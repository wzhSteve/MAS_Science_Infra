#!/usr/bin/env python3
"""Live check of Bing/Wikipedia tools and three MAS stacks on local Qwen3-4B.

Writes a JSON report plus a human-readable log under artifacts/live_mas_bing_tools/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "mas")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "artifacts" / "live_mas_bing_tools"
LOG = OUT / "run.log"
REPORT = OUT / "report.json"
BASE_URL = os.environ.get("LIVE_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("LIVE_MODEL", "Qwen3-4B")

TASK = {
    "id": "live-bing-citibank",
    "question": "Who was the United States president when Citibank was founded?",
    "answer": "James Madison",
    "answers": ["James Madison", "Madison"],
    "source": "hotpotqa",
}

FAIL_MARKERS = (
    "search_unavailable",
    "bing unreachable",
    "wikipedia unreachable",
    "tool was not found",
    "google_search",
    "Google_Search_Tool",
    "retrieved_information:",
)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _bind_local_llm() -> None:
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["AI_ASSISTANT_API_KEY"] = "EMPTY"
    for key in (
        "OPENAI_API_BASE",
        "OPENAI_API_BASE_URL",
        "OPENAI_BASE_URL",
        "AI_ASSISTANT_API_BASE",
    ):
        os.environ[key] = BASE_URL
    for key in ("OPENAI_MODEL", "MODEL", "MODEL_NAME", "MODEL_Name", "AI_ASSISTANT_MODEL", "SERVER_MODEL"):
        os.environ[key] = MODEL
    os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = "1"
    os.environ.pop("SCIENCE_EPC_AW_MOCK", None)


def _tool_failed(text: str) -> bool:
    low = str(text or "").lower()
    return any(m.lower() in low for m in FAIL_MARKERS if m != "google_search") or low.startswith(
        ("error:", "search_unavailable", "bing unreachable", "wikipedia unreachable")
    )


def test_kernels() -> Dict[str, Any]:
    from mas.tools.search import bing_search
    from mas.tools.wikipedia import wikipedia_search

    rows = []
    for name, fn, query in (
        ("bing_search", bing_search, "Citibank founded"),
        ("wikipedia_search", wikipedia_search, "Citibank"),
    ):
        t0 = time.time()
        try:
            out = str(fn(query) or "")
            err = None
        except Exception as exc:  # noqa: BLE001
            out, err = "", f"{type(exc).__name__}: {exc}"
        rows.append({
            "tool": name,
            "query": query,
            "seconds": round(time.time() - t0, 2),
            "ok": bool(out) and not _tool_failed(out) and err is None,
            "error": err,
            "has_url": "URL:" in out,
            "has_desc": any(
                line.startswith("   ") and not line.startswith("   URL:")
                for line in out.splitlines()
            ),
            "has_page_body": "retrieved_information:" in out or out.startswith("url: http"),
            "preview": out[:600],
        })
        _log(f"kernel {name} ok={rows[-1]['ok']} url={rows[-1]['has_url']} desc={rows[-1]['has_desc']}")
        _log("  " + out.replace("\n", "\n  ")[:400])
    return {"kernels": rows}


def _extract_tools(messages: List[Any]) -> List[Dict[str, Any]]:
    tools = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        payload = m.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        kind = str(m.get("kind") or "")
        if kind not in ("tool_result", "tool_invoke") and "output" not in payload:
            continue
        output = str(payload.get("output") or payload.get("content") or "")
        name = str(
            payload.get("tool")
            or payload.get("tool_name")
            or (payload.get("args") or {}).get("tool_name")
            or m.get("src")
            or ""
        )
        if not name and kind != "tool_result":
            continue
        tools.append({
            "kind": kind,
            "src": m.get("src"),
            "tool": name,
            "ok": payload.get("ok"),
            "failed": _tool_failed(output),
            "has_google": "google_search" in output.lower() or "Google_Search_Tool" in output,
            "preview": output[:240],
        })
    return tools


def run_mas(name: str, workflow_path: Path) -> Dict[str, Any]:
    from mas_agent import bind_training_llm_env
    from workflow.archive import Archive, register_archive
    from workflow.memory import MemoryStore
    from workflow.runtime import LLMConfig, run_compiled_episode
    from workflow.spec import load_spec

    t0 = time.time()
    error = None
    raw = None
    spec = None
    try:
        spec = load_spec(str(workflow_path.resolve()))
        cfg = LLMConfig(
            endpoint=BASE_URL,
            model=MODEL,
            temperature=0.2,
            max_turns=6,
            max_tokens=1024,
            max_model_len=8192,
            enabled_tools=list(spec.tools),
        )
        arch = register_archive(Archive())
        mem = MemoryStore()
        with bind_training_llm_env(endpoint=BASE_URL, model=MODEL, api_key="EMPTY"):
            raw = run_compiled_episode(TASK, cfg, arch, spec, mem, None)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        _log(f"{name} raised {error}")
    seconds = round(time.time() - t0, 1)
    messages = list(getattr(raw, "messages", None) or [])
    tools = _extract_tools(messages)
    answer = str(getattr(raw, "final_answer", "") or "")
    failed_tools = [t for t in tools if t.get("failed") or t.get("ok") is False]
    google_leaks = [t for t in tools if t.get("has_google")]
    rec = {
        "name": name,
        "workflow": str(workflow_path),
        "seconds": seconds,
        "error": error or getattr(raw, "error", None),
        "answer": answer[:400],
        "n_messages": len(messages),
        "kinds": [str(m.get("kind") or "") for m in messages if isinstance(m, dict)],
        "tools": tools,
        "failed_tools": failed_tools,
        "google_leaks": google_leaks,
        "ok": error is None and not failed_tools and not google_leaks,
    }
    _log(
        f"mas {name} ok={rec['ok']} answer={rec['answer'][:80]!r} "
        f"tools={len(tools)} failed={len(failed_tools)} google={len(google_leaks)}"
    )
    for t in tools:
        _log(f"  tool {t['tool']} kind={t['kind']} ok={t['ok']} failed={t['failed']} :: {t['preview'][:160]}")
    return rec


def main() -> int:
    from workflow.env_load import load_repo_dotenv

    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    load_repo_dotenv()
    _bind_local_llm()
    _log(f"base_url={BASE_URL} model={MODEL}")

    report: Dict[str, Any] = {"base_url": BASE_URL, "model": MODEL, "task": TASK}
    report.update(test_kernels())
    report["mas"] = [
        run_mas("palette_centralized", ROOT / "mas" / "specs" / "templates" / "centralized.yaml"),
        run_mas("epc_aw_cheap", ROOT / "experiments" / "epc_aw_4b" / "cheap" / "workflow.yaml"),
        run_mas("agentflow", ROOT / "experiments" / "agentflow_flow_grpo_4b" / "workflow.yaml"),
    ]
    kernel_ok = all(r["ok"] for r in report["kernels"])
    mas_ok = all(r["ok"] for r in report["mas"])
    report["ok"] = kernel_ok and mas_ok
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"wrote {REPORT}")
    _log(f"OVERALL ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
