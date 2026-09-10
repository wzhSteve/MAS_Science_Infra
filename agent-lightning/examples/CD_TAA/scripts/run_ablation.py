#!/usr/bin/env python3
"""Run paper-track ablation presets over a question list.

Example:
  python scripts/run_ablation.py \\
    --preset no_capability \\
    --questions experiments/ablation/sample_questions.jsonl \\
    --out experiments/ablation/runs/no_capability_s0 \\
    --offline_memory_dir memory \\
    --max_steps 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Repo root: Agent架构/EPC_AW
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAS.epc_aw.models.ablation import PAPER_PRESETS, list_presets, resolve_ablation
from MAS.epc_aw.solver import construct_solver


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                items.append({"id": str(len(items)), "question": obj})
            else:
                items.append(obj)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="EPC_AW paper-track ablation runner")
    parser.add_argument("--preset", default="full", help=f"One of {sorted(PAPER_PRESETS)}")
    parser.add_argument("--questions", required=True, help="JSONL with question / id fields")
    parser.add_argument("--out", required=True, help="Output directory for this run")
    parser.add_argument("--offline_memory_dir", default=None)
    parser.add_argument("--llm_engine_name", default="gpt-4o")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_time", type=int, default=1200)
    parser.add_argument("--evaluation_mode", action="store_true", default=True)
    parser.add_argument("--no_evaluation_mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional max questions")
    args = parser.parse_args()

    if args.preset not in PAPER_PRESETS and not args.preset.startswith("product_"):
        print(f"Warning: preset '{args.preset}' not in paper set. Known: {list_presets()}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(Path(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    ablation = resolve_ablation(args.preset)
    if args.offline_memory_dir:
        ablation.offline_memory_dir = args.offline_memory_dir

    evaluation_mode = False if args.no_evaluation_mode else args.evaluation_mode

    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=[
            "Base_Generator_Tool",
            "Python_Coder_Tool",
            "Wikipedia_Search_Tool",
            "Web_Search_Tool",
            "Google_Search_Tool",
        ],
        tool_engine=["gpt-4o", "gpt-4o", "Default", "Default"],
        output_types="final,direct",
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        temperature=0.0,
        evaluation_mode=evaluation_mode,
        ablation=ablation,
        offline_memory_dir=args.offline_memory_dir,
    )

    pred_path = out_dir / "predictions.jsonl"
    telem_path = out_dir / "layer_telemetry.jsonl"
    meta = {
        "preset": ablation.preset,
        "ablation": ablation.to_dict(),
        "n_questions": len(questions),
        "started_at": time.time(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pred_path.open("w", encoding="utf-8") as pred_f, telem_path.open(
        "w", encoding="utf-8"
    ) as telem_f:
        for i, item in enumerate(questions):
            qid = str(item.get("id", i))
            question = item.get("question") or item.get("query") or ""
            image = item.get("image") or item.get("file_path")
            t0 = time.time()
            try:
                result = solver.solve(question, image_path=image)
            except Exception as e:
                result = {"error": str(e), "query": question, "task_id": None}
            elapsed = time.time() - t0
            row = {
                "id": qid,
                "question": question,
                "preset": ablation.preset,
                "elapsed_s": round(elapsed, 2),
                "direct_output": result.get("direct_output") if isinstance(result, dict) else None,
                "final_output": result.get("final_output") if isinstance(result, dict) else None,
                "task_id": result.get("task_id") if isinstance(result, dict) else None,
                "error": result.get("error") if isinstance(result, dict) else None,
            }
            pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_f.flush()

            if isinstance(result, dict):
                for ev in result.get("layer_telemetry") or []:
                    telem_f.write(
                        json.dumps({"id": qid, **ev}, ensure_ascii=False) + "\n"
                    )
                telem_f.flush()

            print(f"[{i+1}/{len(questions)}] {qid} done in {elapsed:.1f}s preset={ablation.preset}")

    meta["finished_at"] = time.time()
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {pred_path} and {telem_path}")


if __name__ == "__main__":
    main()
