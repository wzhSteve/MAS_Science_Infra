#!/usr/bin/env python3
"""Offline replay of the FCR planner scoring chain over dumped rollout trees.

Question answered without a GPU: after the lenient prediction anchor + the Rf
scoreability gate, what share of planner steps can produce a p_satisfy, and how
much of the r3 loss was format drift versus lost evidence?

Evidence layers, ranked by trustworthiness (nothing truncated is ever used as
if it were complete):

1. ACTIONS FORENSICS -- primary. Every prediction event stores its own
   ``prediction`` / ``prediction_reason`` / ``p_satisfy`` fields through
   ``context.export_state`` -> execution detail JSON. These fields are NOT
   truncated on disk (sampled files: ``content_truncated=0``). Each event falls
   into exactly one class:
     scored        p_satisfy present (the old chain already handled it)
     anchor_gap    no p, but the step wrote a prediction value or reason ->
                   the JSON was structurally complete; only the prediction
                   token failed the quoted anchor. This is what the lenient
                   anchor rescues.
     no_text       no p, no prediction, no reason -> the completion was
                   truncated before the field or the call itself failed. No
                   anchor can recover text that was never produced; these
                   rows cap what any offline re-score can promise.

2. TEXT REPLAY -- secondary. ``response_head`` in r3 data is capped at 300
   chars (engine.py, pre-fix) and the planner writes ``prediction`` AFTER
   outline/subgoal/tool_input, so the field is visible in ~8% of calls. Within
   that visible window we compare the old strict anchor against the lenient
   anchor; the measured rescue rate extrapolates to the anchor_gap class.
   Rows whose head hit the cap are reported UNJUDGABLE, never counted as
   recovered or lost. Post-fix rollouts carry the uncapped text and replay
   fully.

The plan's pass line (step5/6 >= 0.98, step7 >= 0.85) is evaluated on the
RESCUABLE subset -- ``(scored + anchor_gap)`` events, excluding rows where the
model never produced a prediction before the token cap ("截断顶 cap 的行无法
救，剩余交给 Rf", per plan) -- as
``(scored + rescue_rate x anchor_gap) / (scored + anchor_gap)``. Where r3 data
cannot prove the line (value-cut rows), the authoritative check is the r4
smoke run's in-process ``fcr/pred_parse_rate``.

Usage:
  python3 scripts/rescore_planner_format.py \
    --exec-dir experiments/fcr-step5-to10/artifacts/rollout-trees/executions \
    --events archive/r3_fcr_step5_to10/fcr_events.fcr-step5-to10.jsonl \
    --steps 5,6,7 --out archive/r3_fcr_step5_to10/rescore_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
ADAPTED = ROOT / "user_space" / "projects" / "simple_pev" / "adapted"
if str(ADAPTED) not in sys.path:
    sys.path.insert(0, str(ADAPTED))

from prediction_probs import prediction_anchor  # noqa: E402
from sft_prompts import (  # noqa: E402
    normalize_prediction,
    parse_planner_payload,
    planner_step_scoreable,
)

# The OLD anchor: required a quote right after the colon. Every bare
# "prediction": NO slipped through it into ``except: {}`` in run 9a45bbd456ad.
_STRICT_ANCHOR = re.compile(r'"prediction"\s*:\s*"')
_PRED_VALUE = re.compile(r'"prediction"\s*:\s*("(?:[^"\\]|\\.)*"|[^,\s}\]\[]+)')

LEGACY_HEAD_CAP = 300


def _steps_from_events(path: Path) -> Dict[str, int]:
    """rollout_id -> global_step, from the archived FCR event log."""
    out: Dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("rollout_id") or "")
            step = row.get("global_step")
            if rid and isinstance(step, int):
                out.setdefault(rid, step)
    return out


def _has_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _iter_actions(doc: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    afs = (doc.get("output") or {}).get("agentflow_state") or {}
    actions = afs.get("actions") or {}
    if isinstance(actions, dict):
        for act in actions.values():
            if isinstance(act, dict) and act.get("prediction_event_id"):
                yield act


def _iter_planner_calls(doc: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    trace = (doc.get("output") or {}).get("trace")
    if not isinstance(trace, dict):
        return
    llm = trace.get("llm_calls")
    if not isinstance(llm, dict):
        return
    calls = llm.get("calls") or []
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict) and call.get("role") == "planner":
                yield call


def replay_text(text: str) -> Dict[str, Any]:
    """Classify one planner completion under the old and the new chain."""
    match = _PRED_VALUE.search(text)
    raw = match.group(1).rstrip(",}") if match else None
    label_ok = bool(raw is not None and normalize_prediction(raw.strip('"')))
    strict = bool(_STRICT_ANCHOR.search(text)) and label_ok
    lenient = prediction_anchor(text) is not None and label_ok
    return {
        "strict": strict,
        "lenient": lenient,
        "scoreable": planner_step_scoreable(text),
        "raw_value": raw,
    }


def _rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-dir", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--steps", default="5,6,7")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    wanted = {int(x) for x in args.steps.split(",") if x.strip()}

    rid2step = _steps_from_events(args.events)
    act: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    txt: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    raw_values: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sub in sorted(args.exec_dir.iterdir()):
        if not sub.is_dir():
            continue
        index_path = sub / "index.json"
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        step = rid2step.get(str(index.get("rollout_id") or ""))
        if step is None or step not in wanted:
            continue
        seen: set = set()
        for fjson in sub.glob("*.json"):
            if fjson.name == "index.json":
                continue
            try:
                doc = json.loads(fjson.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for event in _iter_actions(doc):
                eid = str(event.get("prediction_event_id"))
                if eid in seen:
                    continue
                seen.add(eid)
                s = act[step]
                s["events"] += 1
                wrote_value = bool(event.get("prediction")) or bool(event.get("prediction_reason"))
                if _has_number(event.get("p_satisfy")):
                    s["scored"] += 1
                elif wrote_value:
                    s["anchor_gap"] += 1
                    # Sub-type forensics on the untruncated actions fields.
                    # actions.prediction is normalize_prediction OUTPUT:
                    #   YES/NO   -> the payload parsed with a legal label and
                    #               p still failed => token-anchor/echo path,
                    #               exactly what the lenient anchor rescues;
                    #   '' + reason -> the strict reader found no legal label.
                    #               Two indistinguishable causes in legacy
                    #               data: the bare "prediction": NO drift
                    #               (json.loads rejects it, the loose reader
                    #               only keeps quoted strings, so prediction
                    #               lands empty while prediction_reason stays)
                    #               -- rescuable -- or an illegal label like
                    #               partial -- NOT rescuable, Rf's job. The
                    #               split is only estimable from text (layer 2).
                    pred = str(event.get("prediction") or "")
                    if pred in ("YES", "NO"):
                        s["gap_canonical_value"] += 1
                    elif event.get("prediction_reason"):
                        s["gap_empty_value"] += 1
                    else:
                        s["gap_other"] += 1
                else:
                    s["no_text"] += 1
                if event.get("prediction_unscoreable"):
                    s["flagged_unscoreable"] += 1
                if event.get("planner_scoreable"):
                    s["flagged_scoreable"] += 1
            for call in _iter_planner_calls(doc):
                s = txt[step]
                s["calls"] += 1
                full = call.get("response_full")
                head = " ".join(str(call.get("response_head") or "").split())
                text = full if isinstance(full, str) and full else head
                complete = bool(isinstance(full, str) and full) or len(text) < LEGACY_HEAD_CAP
                if complete:
                    s["complete_text"] += 1
                else:
                    s["capped_text"] += 1
                if _PRED_VALUE.search(text) is None:
                    if complete:
                        s["no_prediction_written"] += 1
                    else:
                        s["unjudgable"] += 1
                    continue
                s["prediction_visible"] += 1
                verdict = replay_text(text)
                raw_values[step][repr(verdict["raw_value"])] += 1
                # Truncation-aware judging. Legacy heads cut mid-token, and a
                # cut value proves nothing about what the full text wrote:
                #   - complete text: the verdict is trustworthy, count it;
                #   - capped head, quoted value that never closes ('"N): cut;
                #   - capped head, bare value that is not a canonical label
                #     ('PART' could have grown into 'PARTIAL' or been the
                #     start of nothing): cut;
                #   - capped head with an intact quoted label or a canonical
                #     bare label: the lenient/strict verdict is readable.
                # Value-cut rows are UNJUDGABLE, never counted as lenient
                # failures -- trusting the 300-char window would let the very
                # truncation the fix removes deflate the rescue rate.
                raw = verdict["raw_value"]
                if complete:
                    judged = raw is not None
                elif raw is None:
                    judged = False
                elif raw.startswith('"'):
                    judged = raw.endswith('"')
                else:
                    judged = bool(normalize_prediction(raw))
                if not judged:
                    s["truncated_value_unjudgable"] += 1
                    continue
                # Judged population: prediction VALUE fully visible (complete
                # text, or capped head whose value is intact -- enough for the
                # anchor verdict). The headline rescue rate uses exactly this.
                s["j_total"] += 1

                if verdict["scoreable"]:
                    s["rf_scoreable"] += 1
                if verdict["strict"]:
                    s["j_strict_ok"] += 1
                if verdict["lenient"]:
                    s["j_lenient_ok"] += 1
                if verdict["lenient"] and not verdict["strict"]:
                    s["j_rescued"] += 1
                if not verdict["lenient"]:
                    s["j_still_unscoreable"] += 1

    lines: List[str] = [
        "# Planner format offline rescore",
        "",
        f"- exec dir: `{args.exec_dir}`",
        f"- events: `{args.events}`",
        f"- steps: {sorted(wanted)}",
        "- Layer 1 (actions forensics, untruncated fields) decides the verdict.",
        "- Layer 2 (planner text) only measures the anchor rescue rate; r3 heads",
        "  are legacy 300-char caps, so capped rows are UNJUDGABLE, never assumed.",
        "",
    ]
    failed: List[str] = []
    for step in sorted(set(act) | set(txt)):
        a = act[step]
        t = txt[step]
        events = a["events"] or 0
        # Plan pass line EXCLUDES rows where the model never emitted a
        # prediction before hitting the token cap ("截断顶 cap 的行无法救,
        # 剩余交给 Rf"): those are unrecoverable by any anchor and leave the
        # denominator by design.
        rescuable = (a["scored"] + a["anchor_gap"]) or 0
        ceiling = _rate(a["scored"] + a["anchor_gap"], rescuable)
        measured = _rate(a["scored"], events)
        gap_share = _rate(a["anchor_gap"], events)
        no_text_share = _rate(a["no_text"], events)
        # Headline rescue rate: only texts where the prediction VALUE is fully
        # visible (complete text, or a legacy head whose value survived the
        # cut). Rows whose value was chopped mid-token are UNJUDGABLE and never
        # counted as failures -- counting them would let the 300-char cap the
        # fix removes fake a lower rescue rate.
        j_total = t["j_total"]
        j_rescued = t["j_rescued"]
        j_still = t["j_still_unscoreable"]
        j_misses = j_rescued + j_still
        rescue_rate = _rate(j_rescued, j_misses) if j_misses else 1.0
        # Estimated new-chain score rate: already-scored rows keep working,
        # anchor_gap rows recover at the measured rescue rate.
        est_recovered = _rate(a["scored"] + a["anchor_gap"] * rescue_rate, rescuable)
        # Share over ALL events, what label_coverage will read at run time.
        est_of_all = _rate(a["scored"] + a["anchor_gap"] * rescue_rate, events)
        thr = 0.98 if step in (5, 6) else 0.85
        # The plan line is judged on the rescuable subset: "截断顶 cap 的行无法
        # 救，剩余交给 Rf" excludes no_text rows from the promise by design.
        verdict = "PASS" if est_recovered >= thr else "MISS"
        if verdict != "PASS":
            failed.append(f"step {step} est recovery {est_recovered:.4f} < {thr}")
        lines += [
            f"## step {step} (plan line: >= {thr:.2f}, cap-truncated rows excluded)",
            f"- prediction events: {int(events)} | rescuable (excl. no_text): {int(rescuable)}",
            f"- scored at training time (p present): {int(a['scored'])} ({measured:.4f} of all)",
            f"- anchor_gap (value/reason written, no p): {int(a['anchor_gap'])} ({gap_share:.4f} of all)"
            f" = canonical-label {int(a['gap_canonical_value'])} +"
            f" empty-value-with-reason {int(a['gap_empty_value'])} (bare drift OR illegal label)"
            f" + other {int(a['gap_other'])}",
            f"- no_text (cap-truncated/failed before prediction, excluded per plan): "
            f"{int(a['no_text'])} ({no_text_share:.4f} of all)",
            f"- unscoreable flags on disk (pre-fix rollouts: 0 expected): {int(a['flagged_unscoreable'])}",
            f"- lenient-anchor rescue rate on value-visible texts: {rescue_rate:.4f} "
            f"({int(j_rescued)} rescued / {int(j_misses)} strict misses of {int(j_total)} judged; "
            f"{int(t['truncated_value_unjudgable'])} value-cut rows unjudgable)",
            f"- **estimated new-chain score rate (rescuable subset): {est_recovered:.4f} -> {verdict}**",
            f"- estimated share of ALL events p-scoreable after fix: {est_of_all:.4f}",
            f"- text layer: calls {int(t['calls'])}, complete {int(t['complete_text'])}, "
            f"legacy-capped {int(t['capped_text'])}, field never in window {int(t['unjudgable'])}, "
            f"no prediction in complete text {int(t['no_prediction_written'])}",
            f"- Rf scoreable over value-visible texts: {_rate(t['rf_scoreable'], j_total):.4f}",
            f"- raw prediction values seen: {sorted(raw_values[step].items(), key=lambda x: -x[1])[:8]}",
        ]
        if t["capped_text"]:
            lines.append(
                f"- caveat: {int(t['capped_text']) / (t['calls'] or 1):.1%} of planner texts in this "
                "dump are legacy 300-char heads. engine.py now stores the full response, so r4 "
                "measures pred_parse_rate directly in-process with no sampling caveats."
            )
        lines.append("")

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    if failed:
        print("PLAN LINE MISSED (see caveats): " + "; ".join(failed))
        return 1
    print("ALL STEPS: ceiling meets the plan line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
