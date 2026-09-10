#!/usr/bin/env python3
"""Analyze GAIA ablation logs + per-pid telemetry for module-effectiveness evidence.

Example:
  python scripts/analyze_run_evidence.py \\
    --logs_dir logs \\
    --out logs/ablation_evidence_report.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def analyze_log_text(text: str) -> Dict[str, Any]:
    return {
        "ablation_banner": bool(re.search(r"\[Ablation\] preset=", text)),
        "offline_memory_loaded": "Loaded offline memory" in text,
        "capability_injected": "[MemoryHint] TOOL CAPABILITIES injected" in text,
        "capability_skipped": "[MemoryHint] TOOL CAPABILITIES skipped" in text,
        "invocation_injected": "[MemoryHint] TOOL INVOCATION injected" in text,
        "invocation_skipped": "[MemoryHint] TOOL INVOCATION skipped" in text,
        "causal_diagnosis": "Causal Diagnosis" in text,
        "level_2a": bool(re.search(r"Level 2a", text)),
        "level_2b": bool(re.search(r"Level 2b", text)),
        "level_3": ("Level 3: Counterfactual" in text) or ("Level 3 Counterfactual" in text),
        "intervention_disabled": "Ablation no_intervention" in text or "intervention_disabled" in text,
        "l2a_succeeded": "Level 2a succeeded" in text,
        "l2b_succeeded": "Level 2b succeeded" in text,
        "eval_correct": len(re.findall(r"Correct=True", text)),
        "eval_wrong": len(re.findall(r"Correct=False", text)),
        "pids": [int(x) for x in re.findall(r"\[Evaluating\] pid=(\d+)", text)],
    }


def analyze_telemetry_dir(telem_dir: Path) -> Dict[str, Any]:
    events = Counter()
    pids = []
    correct = 0
    total = 0
    for path in sorted(telem_dir.glob("pid_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pids.append(data.get("pid"))
        total += 1
        if data.get("correct"):
            correct += 1
        for ev in data.get("layer_telemetry") or []:
            events[ev.get("event")] += 1
    l2a_enter = events.get("l2a_enter", 0)
    l2b_enter = events.get("l2b_enter", 0)
    return {
        "pids": pids,
        "n": total,
        "correct": correct,
        "event_counts": dict(events),
        "intervention_trigger": events.get("intervention_trigger", 0),
        "intervention_disabled": events.get("intervention_disabled", 0),
        "l2a_rescue_rate": (events.get("l2a_complete", 0) / l2a_enter) if l2a_enter else None,
        "l2b_rescue_rate": (events.get("l2b_complete", 0) / l2b_enter) if l2b_enter else None,
        "l3_enter": events.get("l3_enter", 0),
    }


def infer_preset_from_name(name: str) -> str:
    for p in (
        "no_intervention",
        "no_capability",
        "no_invocation",
        "no_memory",
        "product_no_l2a",
        "product_no_l2b",
        "product_no_l3",
        "full",
    ):
        if name.endswith(f"_{p}.log") or f"_{p}_" in name or name.endswith(f"/{p}") or f"/{p}_" in name.replace("\\", "/"):
            return p
    m = re.search(r"_(full|no_intervention|no_capability|no_invocation|no_memory)(?:\.log|$)", name)
    return m.group(1) if m else "unknown"


def consistency_checks(preset: str, log_ev: Dict[str, Any], telem: Optional[Dict[str, Any]]) -> List[str]:
    notes = []
    if preset == "full":
        if log_ev.get("capability_skipped"):
            notes.append("FAIL: full but capability skipped")
        if log_ev.get("invocation_skipped") and not log_ev.get("invocation_injected"):
            notes.append("WARN: full run never injected TOOL INVOCATION (may lack executor steps)")
        if telem and telem.get("intervention_disabled", 0) > 0:
            notes.append("FAIL: full but intervention_disabled events present")
    if preset == "no_intervention":
        if telem:
            ec = telem.get("event_counts") or {}
            if ec.get("l2a_enter", 0) > 0:
                notes.append("FAIL: no_intervention but l2a_enter>0")
            if ec.get("l2b_enter", 0) > 0:
                notes.append("FAIL: no_intervention but l2b_enter>0")
            if ec.get("l3_enter", 0) > 0:
                notes.append("FAIL: no_intervention but l3_enter>0")
            if telem.get("intervention_trigger", 0) > 0 and telem.get("intervention_disabled", 0) == 0:
                notes.append("FAIL: incomplete triggers without intervention_disabled")
        if log_ev.get("level_2a") and not log_ev.get("intervention_disabled"):
            notes.append("WARN: log mentions Level 2a under no_intervention")
    if preset == "no_capability":
        if log_ev.get("capability_injected"):
            notes.append("FAIL: no_capability but TOOL CAPABILITIES injected")
        if not log_ev.get("capability_skipped") and log_ev.get("ablation_banner"):
            notes.append("WARN: expected capability skipped marker")
    if preset == "no_invocation":
        if log_ev.get("invocation_injected"):
            notes.append("FAIL: no_invocation but TOOL INVOCATION injected")
        if not log_ev.get("invocation_skipped") and log_ev.get("ablation_banner"):
            notes.append("WARN: expected invocation skipped marker")
    if not notes:
        notes.append("OK: evidence consistent with preset")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--out", default="logs/ablation_evidence_report.md")
    parser.add_argument(
        "--glob",
        default="test_gaia_*_*.log",
        help="Log glob under logs_dir",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_dir.glob(args.glob))
    # Prefer ablation-tagged logs from this plan
    if not log_files:
        log_files = sorted(logs_dir.glob("test_gaia_*.log"))

    sections = ["# Ablation evidence report", ""]
    summary_rows = [
        "| log | preset | Cap | Inv | Intervene | L2a/L2b/L3 | disabled | EM | consistency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for log_path in log_files:
        text = _read_text(log_path)
        log_ev = analyze_log_text(text)
        preset = infer_preset_from_name(log_path.name)
        # Find matching telemetry dir: logs/ablation_runs/{preset}_{timestamp}
        telem = None
        runs_root = logs_dir / "ablation_runs"
        if runs_root.exists():
            # match by timestamp fragment in log name
            # log: test_gaia_YYYYMMDD_HHMMSS_preset.log
            m = re.search(r"test_gaia_(\d{8}_\d{6})_" + re.escape(preset), log_path.name)
            if m:
                cand = runs_root / f"{preset}_{m.group(1)}"
                if cand.exists():
                    telem = analyze_telemetry_dir(cand)
            if telem is None:
                # fallback: newest dir starting with preset_
                cands = sorted(runs_root.glob(f"{preset}_*"), reverse=True)
                if cands:
                    telem = analyze_telemetry_dir(cands[0])

        notes = consistency_checks(preset, log_ev, telem)
        em = "n/a"
        if log_ev["eval_correct"] + log_ev["eval_wrong"] > 0:
            tot = log_ev["eval_correct"] + log_ev["eval_wrong"]
            em = f"{log_ev['eval_correct']}/{tot}"

        ladder = "/".join(
            [
                "Y" if log_ev["level_2a"] else "n",
                "Y" if log_ev["level_2b"] else "n",
                "Y" if log_ev["level_3"] else "n",
            ]
        )
        summary_rows.append(
            f"| `{log_path.name}` | {preset} | "
            f"{'Y' if log_ev['capability_injected'] else ('skip' if log_ev['capability_skipped'] else '?')} | "
            f"{'Y' if log_ev['invocation_injected'] else ('skip' if log_ev['invocation_skipped'] else '?')} | "
            f"{'Y' if log_ev['causal_diagnosis'] else 'n'} | {ladder} | "
            f"{'Y' if log_ev['intervention_disabled'] else 'n'} | {em} | {notes[0]} |"
        )

        sections.append(f"## `{log_path.name}` (preset={preset})")
        sections.append("")
        sections.append(f"- offline_memory_loaded: {log_ev['offline_memory_loaded']}")
        sections.append(f"- capability_injected/skipped: {log_ev['capability_injected']}/{log_ev['capability_skipped']}")
        sections.append(f"- invocation_injected/skipped: {log_ev['invocation_injected']}/{log_ev['invocation_skipped']}")
        sections.append(f"- causal_diagnosis / L2a / L2b / L3: {log_ev['causal_diagnosis']} / {log_ev['level_2a']} / {log_ev['level_2b']} / {log_ev['level_3']}")
        sections.append(f"- intervention_disabled: {log_ev['intervention_disabled']}")
        sections.append(f"- l2a/l2b succeeded (log): {log_ev['l2a_succeeded']} / {log_ev['l2b_succeeded']}")
        if telem:
            sections.append(f"- telemetry events: `{telem.get('event_counts')}`")
            sections.append(f"- L2a/L2b rescue_rate: {telem.get('l2a_rescue_rate')} / {telem.get('l2b_rescue_rate')}")
        sections.append("- consistency:")
        for n in notes:
            sections.append(f"  - {n}")
        sections.append("")

    report = "\n".join(["# Ablation evidence report", "", "## Summary", ""] + summary_rows + [""] + sections[2:])
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
