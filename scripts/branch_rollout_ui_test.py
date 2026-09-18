#!/usr/bin/env python3
"""Branch rollout UI acceptance via Control HTTP APIs (same paths as WebUI).

Covers L1 (sites declaration) + L2 (Collect wiring) by default.
With --train: short train start, poll artifacts, then stop (L3 smoke).

Does not replace manual UI clicks; use with docs/BRANCH_ROLLOUT_UI_TEST.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]


def req(
    base: str,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    url = f"{base.rstrip('/')}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            return resp.status, parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}


def wait_health(base: str, attempts: int = 40) -> None:
    last = ""
    for _ in range(attempts):
        try:
            code, body = req(base, "GET", "/api/health", timeout=2.0)
            if code == 200 and isinstance(body, dict) and body.get("ok"):
                return
            last = f"{code} {body}"
        except Exception as e:
            last = str(e)
        time.sleep(0.25)
    raise SystemExit(f"UI 未就绪: {base}/api/health  ({last})")


def make_after_tool_site(
    *,
    tool_id: str = "execute_python",
    agent_id: str = "hub",
    gate: str = "entropy_delta",
    reward_scheme: str = "scalar_grpo",
    beam: int = 2,
    resume_mode: str = "messages",
    enabled: bool = True,
) -> Dict[str, Any]:
    return {
        "id": f"site_after_tool_{agent_id}_{tool_id}",
        "enabled": enabled,
        "anchor": {"kind": "after_tool", "agent_id": agent_id, "tool_id": tool_id, "edge_id": None},
        "when": "first",
        "nth": 1,
        "gate": {"type": gate, "params": {}},
        "fork": {
            "beam_size": beam,
            "share_observation": True,
            "resume_mode": resume_mode,
            "probe_max_tokens": 128,
        },
        "reward": {"scheme": reward_scheme, "p_plus": 0.8, "k_min": 2, "epsilon_f": 1e-3},
        "priority": 10,
    }


def make_router_workflow(
    base_wf: Dict[str, Any],
    *,
    sampling: Dict[str, Any],
    reward_scheme: str,
    beam: int,
) -> Dict[str, Any]:
    """Planner + RouterSpec routing to tool-agents + blank expert (agent-framework)."""
    wf = dict(base_wf)
    wf["schema_version"] = "0.3"
    wf["topology"] = "graph"
    wf["entry_agent"] = "planner"
    hub = dict(wf.get("hub") or {})
    hub["system_prompt"] = "You are the planner. Route via tool calls."
    wf["hub"] = hub
    tools = list(wf.get("tools") or ["execute_python", "wikipedia_search"])
    wf["agents"] = [
        {"id": "planner", "kind": "planner", "role": "planner", "skills": ["plan"], "tools": [], "trainable": True},
        *[{"id": t, "kind": "tool", "role": "tool", "trainable": False} for t in tools],
        {
            "id": "expert_phys",
            "kind": "blank",
            "role": "agent",
            "skills": ["physics"],
            "system_prompt": "You are a physics expert.",
            "trainable": True,
        },
    ]
    wf["routers"] = [
        {
            "id": "route_main",
            "candidates": [*tools, "expert_phys"],
            "strategy": "llm_choice",
        }
    ]
    wf["edges"] = [
        {"from": "planner", "to": "route_main", "kind": "message"},
    ]
    samp = dict(sampling)
    samp["sites"] = [
        # anchor a site on the router node itself (after_agent_turn @ router id)
        make_after_agent_turn_site(agent_id="route_main", reward_scheme=reward_scheme, beam=beam),
        # and on a routed tool-agent boundary
        make_after_tool_site(tool_id=tools[0], agent_id="planner", reward_scheme=reward_scheme, beam=beam),
    ]
    wf["sampling"] = samp
    return wf


def make_after_agent_turn_site(
    *,
    agent_id: str = "hub",
    gate: str = "entropy_delta",
    reward_scheme: str = "scalar_grpo",
    beam: int = 2,
    enabled: bool = True,
) -> Dict[str, Any]:
    return {
        "id": f"site_after_turn_{agent_id}",
        "enabled": enabled,
        "anchor": {"kind": "after_agent_turn", "agent_id": agent_id, "tool_id": None, "edge_id": None},
        "when": "first",
        "nth": 1,
        "gate": {"type": gate, "params": {}},
        "fork": {
            "beam_size": beam,
            "share_observation": True,
            "resume_mode": "messages",
            "probe_max_tokens": 128,
        },
        "reward": {"scheme": reward_scheme, "p_plus": 0.8, "k_min": 2, "epsilon_f": 1e-3},
        "priority": 10,
    }


def make_after_verifier_site(
    *,
    agent_id: str = "verifier",
    gate: str = "verifier_fail",
    reward_scheme: str = "scalar_grpo",
    beam: int = 2,
    enabled: bool = True,
) -> Dict[str, Any]:
    return {
        "id": f"site_after_verifier_{agent_id}",
        "enabled": enabled,
        "anchor": {"kind": "after_verifier", "agent_id": agent_id, "tool_id": None, "edge_id": None},
        "when": "first",
        "nth": 1,
        "gate": {"type": gate, "params": {}},
        "fork": {
            "beam_size": beam,
            "share_observation": True,
            "resume_mode": "messages",
            "probe_max_tokens": 128,
        },
        "reward": {"scheme": reward_scheme, "p_plus": 0.8, "k_min": 2, "epsilon_f": 1e-3},
        "priority": 10,
    }


def make_pev_workflow(
    base_wf: Dict[str, Any],
    *,
    sampling: Dict[str, Any],
    reward_scheme: str,
    beam: int,
) -> Dict[str, Any]:
    """Minimal planner→executor→verifier graph + trajectory-style sites (UI YAML shape)."""
    pev = dict(base_wf)
    pev["topology"] = "graph"
    pev["entry_agent"] = "planner"
    hub = dict(pev.get("hub") or {})
    hub["verify"] = "verifier"
    pev["hub"] = hub
    pev["agents"] = [
        {"id": "planner", "role": "planner", "skills": ["plan"], "tools": [], "trainable": True},
        {
            "id": "executor",
            "role": "executor",
            "skills": ["react_loop"],
            "tools": list(pev.get("tools") or ["execute_python"]),
            "trainable": True,
        },
        {"id": "verifier", "role": "verifier", "skills": ["verifier"], "tools": [], "trainable": False},
    ]
    pev["edges"] = [
        {"from": "planner", "to": "executor", "kind": "route"},
        {"from": "executor", "to": "verifier", "kind": "message"},
        {"from": "verifier", "to": "planner", "kind": "feedback"},
    ]
    samp = dict(sampling)
    samp["sites"] = [
        make_after_agent_turn_site(agent_id="planner", reward_scheme=reward_scheme, beam=beam),
        make_after_agent_turn_site(agent_id="executor", reward_scheme=reward_scheme, beam=beam),
        make_after_verifier_site(agent_id="verifier", reward_scheme=reward_scheme, beam=beam),
    ]
    pev["sampling"] = samp
    return pev


def make_on_token_site(*, agent_id: str = "hub") -> Dict[str, Any]:
    return {
        "id": f"site_on_token_{agent_id}",
        "enabled": True,
        "anchor": {"kind": "on_token", "agent_id": agent_id, "tool_id": None, "edge_id": None},
        "when": "first",
        "nth": 1,
        "gate": {"type": "entropy_delta", "params": {}},
        "fork": {
            "beam_size": 2,
            "share_observation": True,
            "resume_mode": "token_prefix",
            "probe_max_tokens": 128,
        },
        "reward": {"scheme": "scalar_grpo"},
        "priority": 20,
    }


def find_enabled_sites(wf: Dict[str, Any]) -> List[Dict[str, Any]]:
    sampling = wf.get("sampling") or {}
    sites = sampling.get("sites") or []
    return [s for s in sites if isinstance(s, dict) and s.get("enabled")]


def scan_expansions(root: Path) -> Dict[str, Any]:
    exp_dir = root / "mas" / ".local_expansion"
    out: Dict[str, Any] = {
        "dir": str(exp_dir),
        "n_files": 0,
        "n_with_plans": 0,
        "max_branch_local_count": 0,
        "sample_plan_meta_keys": [],
    }
    if not exp_dir.is_dir():
        return out
    files = list(exp_dir.glob("*.json"))
    out["n_files"] = len(files)
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        plans = list(payload.get("plans") or [])
        blc = int(payload.get("branch_local_count") or 0)
        out["max_branch_local_count"] = max(out["max_branch_local_count"], blc, len(plans))
        if plans:
            out["n_with_plans"] += 1
            meta = dict((plans[0] or {}).get("meta") or {})
            out["sample_plan_meta_keys"] = sorted(meta.keys())
            if meta.get("action_key") is not None:
                out["has_action_key"] = True
            if int(meta.get("resume_boundary") or 0) > 0:
                out["has_resume_boundary"] = True
            if meta.get("reward_scheme"):
                out["reward_scheme"] = meta.get("reward_scheme")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Branch rollout UI acceptance (Control APIs)")
    p.add_argument("--base", default="http://127.0.0.1:8787")
    p.add_argument("--experiment", default="arpo_e2e")
    p.add_argument("--algo", default="arpo", help="arpo|rae|aepo — drives mode + collect/train")
    p.add_argument("--train", action="store_true", help="run short train smoke + artifact scan")
    p.add_argument("--max-wait", type=float, default=25.0, help="seconds to wait before stop when --train")
    p.add_argument("--include-on-token", action="store_true", help="also inject on_token site (L1.3)")
    p.add_argument(
        "--pev-fixture",
        action="store_true",
        help="PUT minimal PEV workflow + multi-agent sites; assert after_agent_turn/after_verifier (then restore hub sites)",
    )
    p.add_argument(
        "--router-fixture",
        action="store_true",
        help="agent-framework: PUT RouterSpec workflow; assert sites anchor on router node (then restore hub workflow)",
    )
    p.add_argument("--tool", default="execute_python")
    args = p.parse_args()
    no_train = not bool(args.train)

    base = args.base
    exp = args.experiment
    algo = str(args.algo or "arpo").lower().strip()
    if algo == "appo":
        algo = "arpo"
    failed = 0

    def ok(name: str, cond: bool, detail: Any = "") -> None:
        nonlocal failed
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"  [{mark}] {name}{('  ' + str(detail)) if detail not in ('', None) else ''}")

    print(f"Branch rollout UI test → {base}  exp={exp}  algo={algo}  train={not no_train}")
    wait_health(base)

    # --- palette ---
    code, palette = req(base, "GET", "/api/mas/palette")
    ok("GET /api/mas/palette", code == 200 and isinstance(palette, dict))
    modes = list((palette or {}).get("sampling_modes") or []) if isinstance(palette, dict) else []
    gates = list((palette or {}).get("gate_types") or []) if isinstance(palette, dict) else []
    ok("palette has rae", "rae" in modes, modes if "rae" in modes else f"{modes} (restart UI: ./run.sh ui --stop && ./run.sh ui --daemon)")
    ok("palette has arpo", "arpo" in modes, modes)
    ok(
        "palette has gate_types",
        len(gates) >= 3,
        gates[:8] if gates else "(empty — restart UI to load gate_types)",
    )

    # --- load workflow ---
    code, bundle = req(base, "GET", f"/api/experiments/{exp}")
    ok(f"GET /api/experiments/{exp}", code == 200 and isinstance(bundle, dict) and "workflow" in (bundle or {}))
    if code != 200 or not isinstance(bundle, dict):
        print(f"\nBranch UI test FAILED ({failed} checks) — cannot load experiment")
        return 1

    wf = dict(bundle.get("workflow") or {})
    sampling = dict(wf.get("sampling") or {})
    sampling["mode"] = algo if algo != "grpo" else "grpo_n"
    if algo == "rae":
        sampling["mode"] = "rae"
    sampling["group_n"] = int(sampling.get("group_n") or 4)
    sampling["beam_size"] = max(2, int(sampling.get("beam_size") or 2))

    reward_scheme = "rae_adjudicate" if algo == "rae" else "scalar_grpo"
    site_tool = make_after_tool_site(
        tool_id=args.tool,
        reward_scheme=reward_scheme,
        beam=int(sampling["beam_size"]),
        resume_mode="messages",
    )
    site_turn = make_after_agent_turn_site(
        agent_id="hub",
        reward_scheme=reward_scheme,
        beam=int(sampling["beam_size"]),
    )
    sites = [site_tool, site_turn]
    if args.include_on_token:
        sites.append(make_on_token_site())
    sampling["sites"] = sites
    wf["sampling"] = sampling

    code, saved_wf = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": wf})
    ok("PUT workflow with sampling.sites", code == 200, f"status={code}")

    code, got = req(base, "GET", f"/api/experiments/{exp}/workflow")
    got_wf = got if isinstance(got, dict) else {}
    # API may wrap as {workflow: ...} or return workflow dict directly
    if "sampling" not in got_wf and isinstance(got_wf.get("workflow"), dict):
        got_wf = got_wf["workflow"]
    got_sampling = dict(got_wf.get("sampling") or {})
    enabled = find_enabled_sites(got_wf if "sampling" in got_wf else {"sampling": got_sampling})
    if not enabled and isinstance(bundle, dict):
        # re-fetch full experiment
        code2, bund2 = req(base, "GET", f"/api/experiments/{exp}")
        if code2 == 200 and isinstance(bund2, dict):
            got_wf = dict(bund2.get("workflow") or {})
            got_sampling = dict(got_wf.get("sampling") or {})
            enabled = find_enabled_sites(got_wf)

    ok("workflow.sampling.mode persisted", str(got_sampling.get("mode") or "").lower() in (algo, "rae", "arpo", "grpo_n"), got_sampling.get("mode"))
    ok("workflow has enabled sites", len(enabled) >= 2, f"n={len(enabled)}")
    kinds = [str((s.get("anchor") or {}).get("kind")) for s in enabled]
    ok(
        "sites include after_tool AND after_agent_turn",
        "after_tool" in kinds and "after_agent_turn" in kinds,
        kinds,
    )
    if enabled:
        s0 = enabled[0]
        ok("site has gate", bool((s0.get("gate") or {}).get("type")), s0.get("gate"))
        if algo == "rae":
            ok(
                "site.reward.scheme=rae_adjudicate",
                any(str((s.get("reward") or {}).get("scheme")) == "rae_adjudicate" for s in enabled),
                [ (s.get("reward") or {}).get("scheme") for s in enabled ],
            )
    if args.include_on_token:
        ok("on_token site present", "on_token" in kinds, kinds)

    # --- optional PEV fixture (trajectory multi-agent YAML shape) ---
    if args.pev_fixture:
        print("  --- pev-fixture ---")
        pev_wf = make_pev_workflow(
            wf,
            sampling=sampling,
            reward_scheme=reward_scheme,
            beam=int(sampling["beam_size"]),
        )
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": pev_wf})
        ok("PUT pev workflow + sites", code == 200, f"status={code}")
        code2, bund_pev = req(base, "GET", f"/api/experiments/{exp}")
        pev_got = dict((bund_pev or {}).get("workflow") or {}) if isinstance(bund_pev, dict) else {}
        pev_enabled = find_enabled_sites(pev_got)
        pev_kinds = [str((s.get("anchor") or {}).get("kind")) for s in pev_enabled]
        pev_agents = {
            str((s.get("anchor") or {}).get("agent_id") or "")
            for s in pev_enabled
            if (s.get("anchor") or {}).get("kind") == "after_agent_turn"
        }
        ok("pev sites include after_agent_turn", "after_agent_turn" in pev_kinds, pev_kinds)
        ok("pev sites include after_verifier", "after_verifier" in pev_kinds, pev_kinds)
        ok(
            "pev after_agent_turn covers planner+executor",
            "planner" in pev_agents and "executor" in pev_agents,
            sorted(pev_agents),
        )
        agent_ids = [str(a.get("id")) for a in (pev_got.get("agents") or []) if isinstance(a, dict)]
        ok("pev agents planner/executor/verifier", set(agent_ids) >= {"planner", "executor", "verifier"}, agent_ids)
        # restore hub dual sites so Collect/Train still target hub_react
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": wf})
        ok("restore hub workflow after pev-fixture", code == 200)

    # --- optional router fixture (agent-framework A2/A5) ---
    if args.router_fixture:
        print("  --- router-fixture ---")
        rt_wf = make_router_workflow(
            wf,
            sampling=sampling,
            reward_scheme=reward_scheme,
            beam=int(sampling["beam_size"]),
        )
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": rt_wf})
        ok("PUT router workflow + router-anchored sites", code == 200, f"status={code}")
        code2, bund_rt = req(base, "GET", f"/api/experiments/{exp}")
        rt_got = dict((bund_rt or {}).get("workflow") or {}) if isinstance(bund_rt, dict) else {}
        # API may wrap as {workflow: ...}
        if "agents" not in rt_got and isinstance(rt_got.get("workflow"), dict):
            rt_got = rt_got["workflow"]
        rt_enabled = find_enabled_sites(rt_got)
        # sites anchored on the router node survive the round trip
        router_anchored = [
            s
            for s in rt_enabled
            if str((s.get("anchor") or {}).get("agent_id") or "") == "route_main"
        ]
        ok("router site persisted (anchor.agent_id=router id)", len(router_anchored) >= 1,
           [str((s.get("anchor") or {}).get("agent_id")) for s in rt_enabled])
        routers = [dict(r) for r in (rt_got.get("routers") or []) if isinstance(r, dict)]
        ok("workflow.routers persisted", len(routers) >= 1, routers)
        if routers:
            ok("router candidates persisted", bool(routers[0].get("candidates")), routers[0].get("candidates"))
        agent_kinds = {
            str(a.get("id")): str(a.get("kind"))
            for a in (rt_got.get("agents") or [])
            if isinstance(a, dict)
        }
        ok("planner + tool-agents + blank expert kinds", 
           agent_kinds.get("planner") == "planner"
           and any(k == "tool" for k in agent_kinds.values())
           and agent_kinds.get("expert_phys") == "blank",
           agent_kinds)
        # restore hub workflow so Collect/Train still target hub_react
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": wf})
        ok("restore hub workflow after router-fixture", code == 200)

    # disk YAML check
    yaml_path = REPO / "experiments" / exp / "workflow.yaml"
    if yaml_path.is_file():
        text = yaml_path.read_text(encoding="utf-8")
        ok("disk workflow.yaml mentions sites", "sites" in text or "after_tool" in text)
        if algo == "rae":
            ok("disk workflow mode rae or rae_adjudicate", "rae" in text.lower())

    # --- collect mock ---
    code, col = req(
        base,
        "POST",
        f"/api/mas/collect?experiment_id={exp}",
        {"mock": True, "n": 1, "algo": algo},
        timeout=90.0,
    )
    ok(
        "POST collect mock",
        code == 200 and isinstance(col, dict) and int((col or {}).get("n") or 0) >= 1,
        f"status={code} body={col if code != 200 else ''}",
    )
    if code == 200 and isinstance(col, dict):
        sig = col.get("train_signal") or {}
        adv = (sig.get("advantage") or {}) if isinstance(sig, dict) else {}
        name = str(adv.get("name") or "").lower()
        ok(f"collect TrainSignal.name={algo}", name == algo, name)
        print("       NOTE: Collect does not tree-branch; L2 wiring only.")

    # --- optional train ---
    if not no_train:
        code, gpus = req(base, "GET", "/api/gpus")
        rec = (gpus.get("recommend") if isinstance(gpus, dict) else None) or {}
        ids = list((rec.get("devices") or {}).get("ids") or [0])
        code, bund = req(base, "GET", f"/api/experiments/{exp}")
        rl = dict((bund or {}).get("rl") or {}) if isinstance(bund, dict) else {}
        rl_patch = dict(rl)
        rl_patch.update(
            {
                "profile": rec.get("profile") or rl.get("profile") or "fast",
                "algo": algo,
                "n_runners": int(rec.get("n_runners") or rl.get("n_runners") or 1),
                "devices": {"ids": ids},
            }
        )
        algo_block = dict(rl_patch.get("algorithm") or {})
        algo_block["tir_algo"] = algo
        algo_block["adv_estimator"] = "grpo"
        tir = dict(algo_block.get("tir") or {})
        tir["expand_in_runner"] = True
        tir["ready_batch"] = True
        algo_block["tir"] = tir
        rl_patch["algorithm"] = algo_block
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/rl", {"data": rl_patch})
        ok("PUT rl before train", code == 200)

        code, train = req(
            base,
            "POST",
            f"/api/rl/train?experiment_id={exp}",
            {"stop_llm": True, "confirm_gpu": True},
            timeout=30.0,
        )
        ok("POST /api/rl/train", code == 200 and isinstance(train, dict) and train.get("run_id"), train if code != 200 else "")
        run_id = str((train or {}).get("run_id") or "") if isinstance(train, dict) else ""
        if run_id:
            argv = [str(x) for x in ((train or {}).get("argv") or [])]
            ok("train argv --algo", "--algo" in argv and algo in argv, argv)
            t_end = time.time() + float(args.max_wait)
            while time.time() < t_end:
                time.sleep(2.0)
                exp_scan = scan_expansions(REPO)
                if exp_scan["n_with_plans"] > 0 or exp_scan["max_branch_local_count"] > 0:
                    break
                # also peek run log for branch hints
                _, run = req(base, "GET", f"/api/runs/{run_id}?tail=80")
                log = ""
                if isinstance(run, dict):
                    log = str(run.get("log_tail") or run.get("tail") or run.get("log") or "")
                if any(k in log.lower() for k in ("resume_messages", "arpo_branch", "branch", "enqueue")):
                    break
            exp_scan = scan_expansions(REPO)
            print(f"       expansion scan: {exp_scan}")
            branchish = (
                exp_scan["n_with_plans"] > 0
                or exp_scan["max_branch_local_count"] > 0
                or exp_scan["n_files"] > 0
            )
            ok(
                "L3 expansion or branch artifact (soft)",
                branchish,
                "no .local_expansion yet — train may need longer / GPU; check logs",
            )
            # rl.yaml after sync
            rl_path = REPO / "experiments" / exp / "rl.yaml"
            if rl_path.is_file():
                rl_text = rl_path.read_text(encoding="utf-8")
                ok(f"disk rl.yaml tir_algo~={algo}", algo in rl_text.lower() or "tir_algo" in rl_text, "see rl.yaml")
            if algo == "rae" and exp_scan.get("has_action_key"):
                ok("plan meta has action_key", True)
            if exp_scan.get("has_resume_boundary"):
                ok("plan meta has resume_boundary>0", True)

            code, stopped = req(base, "POST", f"/api/rl/stop?experiment_id={exp}", {})
            ok("POST /api/rl/stop", code == 200, stopped)

    if failed:
        print(f"\nBranch UI test FAILED ({failed} checks)")
        print("See docs/BRANCH_ROLLOUT_UI_TEST.md for manual L3 criteria / false-green notes.")
        return 1
    print("\nBranch UI test OK")
    print("Manual: docs/BRANCH_ROLLOUT_UI_TEST.md + docs/ROLLOUT_SAMPLING_UI_TEST.md (S0–S5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
