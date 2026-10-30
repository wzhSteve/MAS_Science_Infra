"""Rescore finished scale256 logs with the local llm_binary gates.

The judge is mocked to 1 so the report shows what the gates reject on their
own. This does not rerun the MAS. The logs were produced by the old process.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
for path in (str(ROOT), str(ROOT / "mas"), str(ROOT / "rl")):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.rewards.outcome import extract_answer_text, score_llm_binary  # noqa: E402

REPORT = ROOT / "user_space" / "projects" / "pure_pev" / "artifacts" / "reward_rescore.json"
LOGS = {
    "grpo": ROOT / "experiments/scale256-grpo/artifacts/runs/42a9079f7cae/stdout.log",
    "flow": ROOT / "experiments/scale256-flow-grpo/artifacts/runs/ba2b145a890c/stdout.log",
    "arpo": ROOT / "experiments/scale256-arpo/artifacts/runs/470b607c398a/stdout.log",
    "appo": ROOT / "experiments/scale256-appo/artifacts/runs/c2af4887cc6c/stdout.log",
}


def _load_lines(path: Path) -> List[str]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    return text.splitlines()


def _wrap_col(lines: List[str]) -> int:
    counts = Counter(len(line.rstrip()) for line in lines if line.startswith(" ") and len(line) >= 80)
    return counts.most_common(1)[0][0] if counts else 80


def _unwrap(buf: List[str], col: int) -> str:
    parts: List[Tuple[str, bool]] = []
    for raw in buf:
        if not raw.startswith(" "):
            continue
        stripped = raw.rstrip(" ")
        lead = len(raw) - len(raw.lstrip(" "))
        body = stripped[lead:]
        parts.append((body, len(stripped) >= col))
    out: List[str] = []
    for index, (piece, hard) in enumerate(parts):
        out.append(piece)
        if not hard and index + 1 < len(parts):
            nxt = parts[index + 1][0]
            if piece and nxt and piece[-1].isalnum() and nxt[0].isalnum():
                out.append(" ")
    return "".join(out)


def _records(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    lines = _load_lines(path)
    col = _wrap_col(lines)
    rewards: Dict[str, str] = {}
    questions: Dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if "[Rollout ro-" in line and ("lit_tir_agent.py:274" in line or "lit_tir_agent.py:132" in line):
            rid = re.search(r"\[Rollout (ro-[0-9a-f]+)\]", line).group(1)
            kind = "R" if ":274" in line else "Q"
            buf: List[str] = []
            index += 1
            while index < len(lines):
                raw = lines[index]
                if (
                    not raw.startswith(" ")
                    or "INFO" in raw
                    or "AgentOps" in raw
                    or "TaskRunner" in raw
                    or "Patched" in raw
                ):
                    break
                buf.append(raw)
                index += 1
                if kind == "R" and re.search(r"t=[\d.]+s$", _unwrap(buf, col)):
                    break
            blob = _unwrap(buf, col)
            if kind == "R":
                rewards[rid] = blob
            else:
                questions.setdefault(rid, blob)
            continue
        index += 1
    return rewards, questions


def _parse_reward(blob: str) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"pred=(.*?)\s*gt=(.*?)\s*fmt=(True|False)\s*R=([0-9.]+)",
        blob,
    )
    if not match:
        return None
    try:
        pred = ast.literal_eval(match.group(1))
        gold = ast.literal_eval(match.group(2))
    except (SyntaxError, ValueError):
        return None
    return {"pred": str(pred), "gt": str(gold), "old_r": float(match.group(4))}


def _question(blob: str) -> str:
    match = re.search(r"q=(.*)$", blob, re.S)
    return match.group(1).strip() if match else ""


def _qkey(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (question or "").lower())[:48]


def _full_questions() -> Dict[str, str]:
    """Logged questions are clipped at 180 characters, so MCQ options get cut."""
    import pandas as pd

    found: Dict[str, str] = {}
    frames = [
        ROOT / "data/rl_datasets/train_10k.parquet",
        ROOT / "data/rl_datasets/valid.parquet",
    ]
    for path in frames:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["extra_info"])
        for extra in frame["extra_info"]:
            if not isinstance(extra, dict):
                continue
            question = str(extra.get("question") or "")
            if question:
                found.setdefault(_qkey(question), question)
    return found


def _bucket(pred: str, gold: str, question: str) -> str:
    low = pred.lower()
    if "<think" in low and ("</think>" not in low or low.strip().startswith("<answer><think") or low.strip().startswith("<think")):
        if extract_answer_text(pred) is None:
            return "think"
    if re.search(r"<answer>\s*short(?:\s+answer)?\s*</answer>", pred, re.IGNORECASE):
        return "short_answer"
    extracted = extract_answer_text(pred) or ""
    if "distance_between_centers" in extracted or "distance_between_centers" in pred:
        return "formula"
    if extracted == "Mats Hummel":
        return "partial_name"
    if extracted == "Mats Hummels":
        return "full_surname"
    if re.fullmatch(r"\$*\s*C\s*\$*", extracted) and "251" in question.replace(" ", "").replace(",", ""):
        return "option_c"
    return ""


def main() -> None:
    calls = {"n": 0}

    def _judge(*_args: Any, **_kwargs: Any) -> float:
        calls["n"] += 1
        return 1.0

    summary: Dict[str, Any] = {"runs": {}, "flipped": [], "required": {}}
    required_rows: Dict[str, List[Dict[str, Any]]] = {
        "think": [],
        "short_answer": [],
        "formula": [],
        "partial_name": [],
        "full_surname": [],
        "option_c": [],
    }
    full_questions = _full_questions()
    with mock.patch("rl.rewards.llm_judge.llm_judge_correct", side_effect=_judge):
        for name, path in LOGS.items():
            rewards, questions = _records(path)
            parsed = 0
            old_one = 0
            new_one = 0
            flipped = 0
            judge_on_old_one = 0
            for rid, blob in rewards.items():
                row = _parse_reward(blob)
                if row is None:
                    continue
                parsed += 1
                question = _question(questions.get(rid, ""))
                question = full_questions.get(_qkey(question), question)
                before = calls["n"]
                new_r = float(score_llm_binary(row["pred"], row["gt"], question=question))
                called = calls["n"] > before
                if row["old_r"] >= 1.0:
                    old_one += 1
                    if called:
                        judge_on_old_one += 1
                if new_r >= 1.0:
                    new_one += 1
                if row["old_r"] >= 1.0 and new_r < 1.0:
                    flipped += 1
                    if len(summary["flipped"]) < 40:
                        summary["flipped"].append(
                            {
                                "run": name,
                                "rid": rid,
                                "gt": row["gt"][:80],
                                "pred_head": re.sub(r"\s+", " ", row["pred"])[:160],
                                "judge_called": called,
                            }
                        )
                kind = _bucket(row["pred"], row["gt"], question)
                if kind and row["old_r"] >= 1.0:
                    required_rows[kind].append({"run": name, "new_r": new_r, "judge_called": called, "gt": row["gt"][:40]})
            summary["runs"][name] = {
                "parsed": parsed,
                "old_r1": old_one,
                "new_r1_with_judge_mocked_true": new_one,
                "flipped_1_to_0": flipped,
                "judge_calls_on_old_r1": judge_on_old_one,
            }
    for kind, rows in required_rows.items():
        summary["required"][kind] = {
            "n": len(rows),
            "new_zero": sum(1 for row in rows if row["new_r"] < 1.0),
            "new_one": sum(1 for row in rows if row["new_r"] >= 1.0),
            "judge_called": sum(1 for row in rows if row["judge_called"]),
        }
    must_zero = ("think", "short_answer", "formula", "partial_name")
    must_one = ("full_surname", "option_c")
    failures = []
    for kind in must_zero:
        block = summary["required"][kind]
        if block["n"] < 1 or block["new_zero"] != block["n"] or block["judge_called"]:
            failures.append(kind)
    for kind in must_one:
        block = summary["required"][kind]
        if block["n"] < 1 or block["new_one"] != block["n"]:
            failures.append(kind)
    summary["ok"] = not failures
    summary["failures"] = failures
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "failures": failures, "required": summary["required"], "runs": summary["runs"]}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
