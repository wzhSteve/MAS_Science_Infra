"""Upload / list / apply user-space projects (management API helpers)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import UploadFile

from science_infra.control.paths import mas_root

SHARED_TOOLS = [
    "Base_Generator_Tool",
    "Python_Coder_Tool",
    "Wikipedia_Search_Tool",
    "Bing_Search_Tool",
    "Web_Fetch_Tool",
]
_REJECTED_TOOLS = frozenset({"Google_Search_Tool", "Ground_Google_Search_Tool"})

# ---------------------------------------------------------------------------
# Fix F-1: flow-doc (project flow documentation from each project's upload/ tree)
# ---------------------------------------------------------------------------

FLOW_DOC_SOURCES: Dict[str, Dict[str, str]] = {
    "agentflow": {
        # Official prompts are f-strings inside the model classes.
        "source_root": "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow",
        "prompts_rel": "models",
        "prompt_files": "planner.py,executor.py,verifier.py",
        "title": "AgentFlow",
        "diagnosis_method": "",
    },
    "epc-aw-main": {
        "source_root": "user_space/projects/epc-aw-main/upload/EPC-AW-main",
        "prompts_rel": "MAS/epc_aw/prompts",
        "title": "EPC-AW",
        "diagnosis_method": "epc_aw_diagnosis",
    },
}

# runtime 是否保留 BTS：agentflow 的 user_space 运行时是纯
# planner→executor→verifier 循环（单计划生成，不触发 BTS）；
# epc-aw-main 的 _select_plan 运行时保留 BTS。
FLOW_DOC_RUNTIME_USES_BTS: Dict[str, bool] = {
    "epc-aw-main": True,
}

AGENTFLOW_AGENTS: List[Dict[str, str]] = [
    {"id": "planner", "label": "规划器"},
    {"id": "executor", "label": "执行器"},
    {"id": "verifier", "label": "校验器"},
]


def _agentflow_steps() -> List[Dict[str, Any]]:
    """Official Solver.solve loop (upload/AgentFlow-main/agentflow/agentflow/solver.py)."""
    rows = [
        ("planner", "analyze_query", "models/planner.py", "第 1 步前分析问题（冻结引擎 llm_engine_fixed）"),
        ("planner", "generate_next_step", "models/planner.py", "按 Memory 选下一个工具与子目标（唯一可训调用）"),
        ("planner", "extract_context_subgoal_and_tool", "", "解析 Context / Sub-Goal / Tool Name；工具不存在时走官方 not-found 分支"),
        ("executor", "generate_tool_command", "models/executor.py", "生成 tool.execute(...) 命令（冻结引擎）"),
        ("executor", "execute_tool_command", "", "执行命令，结果经 make_json_serializable_truncated"),
        ("executor", "memory.add_action", "", "写入唯一共享 Memory"),
        ("verifier", "verificate_context", "models/verifier.py", "判断 Memory 是否足够（冻结引擎）；Conclusion: STOP 跳出主循环"),
        ("verifier", "extract_conclusion", "", "解析 STOP / CONTINUE；max_steps 或 max_time 用尽也结束"),
        ("planner", "generate_direct_output", "models/planner.py", "终答（冻结引擎），结尾 **Answer:** X"),
    ]
    return [
        {"seq": i, "agent": a, "method": m, "prompt": p, "desc": d, "runtime_active": True}
        for i, (a, m, p, d) in enumerate(rows, 1)
    ]

FLOW_AGENTS: List[Dict[str, str]] = [
    {"id": "planner", "label": "规划器"},
    {"id": "executor", "label": "执行器"},
    {"id": "diagnoser/verifier", "label": "诊断校验器"},
]


def _flow_steps(diagnosis_method: str, runtime_uses_bts: bool = True) -> List[Dict[str, Any]]:
    """Solver main-loop steps, in source order (upload/*/solver.py).

    runtime_active=False marks steps that exist in source but are never hit at
    runtime: the agentflow user_space runtime runs a plain planner→executor→
    verifier loop (single-plan generation, never triggering BTS), while
    epc-aw-main keeps BTS inside ``_select_plan``.
    """
    diag = diagnosis_method
    bts_inactive = not runtime_uses_bts
    inactive_note = (
        "（源码含 BTS 代码，但 user_space 运行时为纯 planner→executor→verifier 循环——单计划生成，不触发 BTS）"
        if bts_inactive else ""
    )
    return [
        {
            "seq": 1,
            "agent": "planner",
            "method": "analyze_query",
            "prompt": "planner/analyze_query.txt",
            "desc": "分析问题，生成执行大纲，写入 system_memory",
            "runtime_active": True,
        },
        {
            "seq": 2,
            "agent": "planner",
            "method": "generate_next_step",
            "prompt": "planner/generate_next_step.txt",
            "desc": "按大纲生成下一步候选计划（n>1 时返回候选列表，附 feasibility score）",
            "runtime_active": True,
        },
        {
            "seq": 3,
            "agent": "planner/executor/diagnoser",
            "method": "report_generation ×3 + belief_prediction ×3",
            "prompt": "{planner,executor,diagnoser}/report_generation.txt + belief_prediction.txt",
            "desc": "BTS 六路评分链：三方各自对本方候选打分（report_generation）并预测另两方评分（belief_prediction）"
            + inactive_note,
            "runtime_active": not bts_inactive,
        },
        {
            "seq": 4,
            "agent": "planner",
            "method": "compute_bts_for_plans",
            "prompt": "",
            "desc": "综合九组分数计算 BTS 得分，选出最终执行计划（planner 选案 ≠ BTS 选案时触发后续诊断）"
            + inactive_note,
            "runtime_active": not bts_inactive,
        },
        {
            "seq": 5,
            "agent": "planner",
            "method": "extract_context_subgoal_and_tool",
            "prompt": "",
            "desc": "解析选中计划：context / sub_goal / tool_name（none → Base_Generator_Tool）",
            "runtime_active": True,
        },
        {
            "seq": 6,
            "agent": "executor",
            "method": "generate_tool_command",
            "prompt": "executor/generate_tool_command.txt",
            "desc": "生成工具命令（analysis / explanation / command）；迁移新增硬约束：上一轮 search 返回 SERP 列表时本轮强制 Web_Fetch 第一条 URL",
            "runtime_active": True,
        },
        {
            "seq": 7,
            "agent": "executor",
            "method": "execute_tool_command",
            "prompt": "",
            "desc": "执行工具命令并截断序列化结果",
            "runtime_active": True,
        },
        {
            "seq": 8,
            "agent": "diagnoser/verifier",
            "method": "verificate_context",
            "prompt": "diagnoser/verificate_context.txt",
            "desc": "校验结果是否满足大纲目标；含 STOP 分支：conclusion=STOP（EPC-AW 还要求新增信息且 step>1）则跳出主循环",
            "runtime_active": True,
        },
        {
            "seq": 9,
            "agent": "diagnoser/verifier",
            "method": diag,
            "prompt": f"diagnoser/{diag}.txt",
            "desc": "仅当 planner 选案 ≠ BTS 选案：伪造策略诊断，epistemic_constraint 写入 planner.memory",
            "runtime_active": True,
        },
        {
            "seq": 10,
            "agent": "diagnoser/verifier",
            "method": "update_outline",
            "prompt": "diagnoser/update_outline.txt",
            "desc": "依据本轮结果更新执行大纲（system_memory.set_outline），并记录 last_step_plan_scores",
            "runtime_active": True,
        },
        {
            "seq": 11,
            "agent": "planner/executor/diagnoser",
            "method": "memory.add_action ×三方",
            "prompt": "",
            "desc": "三方 memory 条件写入：有新信息→diagnoser.memory（+add_obtained_information），无新信息→planner.memory，恒→executor.memory",
            "runtime_active": True,
        },
        {
            "seq": 12,
            "agent": "executor",
            "method": "generate_direct_output",
            "prompt": "executor/direct_output.txt",
            "desc": "终答生成（direct_output prompt 要求按标签包裹答案）",
            "runtime_active": True,
        },
    ]


def get_project_flow_doc(project_id: str) -> Dict[str, Any]:
    """Return agents / steps / prompts documentation for an upload-backed project."""
    paths, registry, _s, _v, _a = _gw()
    known = {
        str(row.get("id") or "") for row in registry.list_projects()
    }
    if project_id not in known or project_id not in FLOW_DOC_SOURCES:
        raise FileNotFoundError(f"用户项目不存在或无流程文档: {project_id}")
    cfg = FLOW_DOC_SOURCES[project_id]
    row = registry.get_project(project_id)
    prompts_root = mas_root().parent / cfg["source_root"] / cfg["prompts_rel"]
    prompts: List[Dict[str, Any]] = []
    only = [x for x in str(cfg.get("prompt_files") or "").split(",") if x]
    if prompts_root.is_dir():
        paths = [prompts_root / x for x in only if (prompts_root / x).exists()] if only else sorted(prompts_root.rglob("*.txt"))
        for path in paths:
            agent = path.stem if only else path.parent.name
            item: Dict[str, Any] = {
                "agent": agent,
                "name": f"{agent}/{path.name}",
                "path": f"{cfg['source_root']}/{cfg['prompts_rel']}/{agent}/{path.name}",
                "content": "",
            }
            try:
                item["content"] = path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"读取失败: {exc}"
            prompts.append(item)
    return {
        "project_id": project_id,
        "title": str(row.get("title") or cfg["title"]),
        "source_root": cfg["source_root"],
        "agents": [dict(agent) for agent in (AGENTFLOW_AGENTS if project_id == "agentflow" else FLOW_AGENTS)],
        "steps": _agentflow_steps() if project_id == "agentflow" else _flow_steps(
            cfg["diagnosis_method"],
            runtime_uses_bts=FLOW_DOC_RUNTIME_USES_BTS.get(project_id, True),
        ),
        "prompts": prompts,
    }


DEFAULT_EPC_AW_RUNTIME: Dict[str, Any] = {
    "n": 1,
    "max_steps": 20,
    "max_time": 3000,
    "max_tokens": 4000,
    "temperature": 0.0,
    "enabled_tools": list(SHARED_TOOLS),
}

DEFAULT_AGENTFLOW_RUNTIME: Dict[str, Any] = {
    "n": 1,
    "max_steps": 10,
    "max_time": 300,
    "max_tokens": 2048,
    "temperature": 0.7,
    "memory_result_chars": 0,
    "enabled_tools": list(SHARED_TOOLS),
}


def _gw():
    import sys

    mas = str(mas_root())
    if mas not in sys.path:
        sys.path.insert(0, mas)
    from workflow.user_gateway import apply as apply_mod
    from workflow.user_gateway import paths, registry, scaffold, validate

    return paths, registry, scaffold, validate, apply_mod


def list_user_projects(experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    _paths, registry, _s, _v, _a = _gw()
    return registry.list_projects(experiment_id=experiment_id)


def get_user_project(project_id: str) -> Dict[str, Any]:
    _paths, registry, _s, validate, _a = _gw()
    data = registry.get_project(project_id)
    data["validation"] = validate.validate_project(project_id)
    return data


def _family_for_project(project_id: str, title: str = "", wraps: str = "") -> str:
    from workflow.user_gateway.detect import detect_mas_family

    try:
        family = detect_mas_family(project_id, title, wraps=wraps)
    except Exception:
        family = ""
    if family:
        return family
    wraps_l = wraps.lower()
    if wraps_l.startswith("agentflow") or "agentflow" in wraps_l:
        return "agentflow"
    if wraps_l.startswith("epc-aw") or wraps_l.startswith("epc_aw"):
        return "epc_aw"
    return ""


def _default_runtime_for_family(family: str) -> Dict[str, Any]:
    if family == "agentflow":
        return dict(DEFAULT_AGENTFLOW_RUNTIME)
    return dict(DEFAULT_EPC_AW_RUNTIME)


def _allowed_tools_for_family(family: str) -> List[str]:
    names = (
        list(DEFAULT_AGENTFLOW_RUNTIME.get("enabled_tools") or SHARED_TOOLS)
        if family == "agentflow"
        else list(SHARED_TOOLS)
    )
    return [name for name in names if name not in _REJECTED_TOOLS]


def _with_allowed_tools(cfg: Dict[str, Any], *, family: str) -> Dict[str, Any]:
    """Read-only overlay for the UI; never persist ``allowed_tools`` to yaml."""
    out = dict(cfg)
    out.pop("allowed_tools", None)
    out["allowed_tools"] = _allowed_tools_for_family(family)
    return out


def palette_user_projects() -> List[Dict[str, Any]]:
    _paths, registry, _s, validate, _a = _gw()

    items = []
    for row in registry.list_projects():
        pid = str(row.get("id") or "")
        if not pid:
            continue
        wf = {}
        try:
            wf = validate.load_project_workflow(pid)
        except Exception:
            wf = {}
        wraps = str(row.get("wraps") or "")
        family = _family_for_project(pid, str(row.get("title") or ""), wraps)
        items.append(
            {
                "id": pid,
                "title": row.get("title") or pid,
                "mode": row.get("mode") or "io_module",
                "status": row.get("status") or "uploaded",
                "agent_ids": list(row.get("agent_ids") or []),
                "tool_ids": list(row.get("tool_ids") or []),
                "workflow": wf or None,
                "family": family,
                "wraps": wraps,
            }
        )
    return items


def _runtime_path(project_id: str) -> Path:
    paths, _r, _s, _v, _a = _gw()
    return paths.project_dir(project_id) / "contracts" / "runtime.yaml"


def _clamp_runtime(data: Dict[str, Any], *, family: str = "") -> Dict[str, Any]:
    defaults = _default_runtime_for_family(family)
    cfg = dict(defaults)
    cfg["enabled_tools"] = list(defaults.get("enabled_tools") or SHARED_TOOLS)
    if not isinstance(data, dict):
        return cfg
    try:
        cfg["n"] = max(1, min(16, int(data.get("n", cfg["n"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("n 必须是 1–16 的整数") from exc
    try:
        cfg["max_steps"] = max(1, min(40, int(data.get("max_steps", cfg["max_steps"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_steps 必须是 1–40 的整数") from exc
    try:
        cfg["max_time"] = max(30, min(10000, int(data.get("max_time", cfg["max_time"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_time 必须是 30–10000 的整数") from exc
    try:
        cfg["max_tokens"] = max(256, min(16384, int(data.get("max_tokens", cfg["max_tokens"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens 必须是 256–16384 的整数") from exc
    try:
        cfg["temperature"] = float(data.get("temperature", cfg["temperature"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature 必须是数字") from exc
    if cfg["temperature"] < 0 or cfg["temperature"] > 2:
        raise ValueError("temperature 必须在 0–2")
    tools = data.get("enabled_tools", cfg["enabled_tools"])
    if not isinstance(tools, list) or not tools:
        raise ValueError("enabled_tools 至少勾选一项")
    allowed = set(_allowed_tools_for_family(family))
    cleaned = [
        str(item)
        for item in tools
        if str(item) in allowed and str(item) not in _REJECTED_TOOLS
    ]
    if not cleaned:
        raise ValueError("enabled_tools 无效")
    cfg["enabled_tools"] = cleaned
    cfg.pop("allowed_tools", None)
    if "memory_result_chars" in cfg:
        try:
            cfg["memory_result_chars"] = max(0, int(data.get("memory_result_chars", cfg["memory_result_chars"]) or 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("memory_result_chars 必须是非负整数") from exc
    return cfg


def get_project_runtime(project_id: str) -> Dict[str, Any]:
    paths, registry, _s, _v, _a = _gw()
    row = registry.get_project(project_id)  # raises if missing
    family = _family_for_project(
        project_id, str(row.get("title") or ""), str(row.get("wraps") or "")
    )
    path = _runtime_path(project_id)
    raw: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    return _with_allowed_tools(_clamp_runtime(raw, family=family), family=family)


def put_project_runtime(project_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    paths, registry, _s, _v, _a = _gw()
    row = registry.get_project(project_id)
    family = _family_for_project(
        project_id, str(row.get("title") or ""), str(row.get("wraps") or "")
    )
    cfg = _clamp_runtime(data or {}, family=family)
    path = _runtime_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    persist = {k: v for k, v in cfg.items() if k != "allowed_tools"}
    path.write_text(
        yaml.safe_dump(persist, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return _with_allowed_tools(cfg, family=family)


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"非法 zip 路径: {name}")
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"zip 路径逃逸: {name}")
    zf.extractall(dest)


def ingest_upload(
    *,
    title: str,
    mode: str,
    experiment_id: str = "",
    project_id: Optional[str] = None,
    upload: Optional[UploadFile] = None,
    raw: Optional[bytes] = None,
    filename: str = "",
) -> Dict[str, Any]:
    paths, registry, scaffold, _validate, _apply = _gw()
    paths.ensure_user_space()
    pid = project_id or paths.slugify_project_id(title)
    try:
        manifest = registry.create_project(pid, title=title, mode=mode, experiment_id=experiment_id)
    except FileExistsError:
        suffix = 2
        base = pid
        while True:
            cand = f"{base}-{suffix}"[:48]
            try:
                manifest = registry.create_project(cand, title=title, mode=mode, experiment_id=experiment_id)
                pid = cand
                break
            except FileExistsError:
                suffix += 1
    root = paths.project_dir(pid)
    upload_dir = root / "upload"
    payload = raw
    name = filename
    if upload is not None:
        payload = payload or upload.file.read()
        name = name or (upload.filename or "upload.bin")
    if payload:
        dest = upload_dir / Path(name).name
        if name.lower().endswith(".zip"):
            _safe_extract(zipfile.ZipFile(io.BytesIO(payload)), upload_dir)
        else:
            dest.write_bytes(payload)
    built = scaffold.scaffold_project(pid, mode if mode in ("io_module", "native_mas") else "io_module", title=title)
    manifest = registry.load_manifest(pid)
    return {"project": manifest, "scaffold": {k: v for k, v in built.items() if k != "workflow"}, "workflow": built.get("workflow")}


def apply_project(project_id: str, current: Optional[Dict[str, Any]] = None, *, replace: bool = False) -> Dict[str, Any]:
    _p, registry, scaffold, _v, apply_mod = _gw()
    manifest = registry.get_project(project_id)
    from workflow.user_gateway.detect import detect_mas_family

    family = detect_mas_family(
        project_id,
        str(manifest.get("title") or ""),
        wraps=str(manifest.get("wraps") or ""),
    )
    want_native = replace or manifest.get("mode") == "native_mas"
    if want_native:
        if family == "epc_aw":
            from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev

            scaffold_epc_aw_pev(project_id, title=str(manifest.get("title") or ""))
            manifest = registry.get_project(project_id)
    merged = apply_mod.merge_user_workflow(current or {}, project_id, replace=replace or manifest.get("mode") == "native_mas")
    return {"workflow": merged, "mode": manifest.get("mode"), "project": manifest}


def list_project_files(project_id: str) -> List[Dict[str, str]]:
    paths, _r, _s, _v, _a = _gw()
    root = paths.project_dir(project_id)
    out: List[Dict[str, str]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        zone = rel.split("/", 1)[0] if "/" in rel else rel
        writable = zone in paths.WRITABLE_REL_DIRS
        out.append({"path": rel, "zone": zone, "writable": "true" if writable else "false"})
    return out


def read_project_file(project_id: str, rel: str) -> str:
    paths, _r, _s, _v, _a = _gw()
    path = paths.assert_in_user_space(paths.project_dir(project_id) / rel, must_exist=True)
    return path.read_text(encoding="utf-8")
