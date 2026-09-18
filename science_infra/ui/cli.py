"""Thin CLI: collect / diagnose / status / dashboard / doctor. Does not hold a graph."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
TIR = ROOT / "mas"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from science_infra.env import env_loaded_from, load_science_env

load_science_env()


def _ensure_tir_path() -> None:
    p = str(TIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_tasks(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return [
            {"id": "demo-1", "question": "What is 1+1?", "answer": "2", "source": "gsm8k", "_mock_answer": "2"},
            {"id": "demo-2", "question": "What is 2+2?", "answer": "4", "source": "gsm8k", "_mock_answer": "4"},
        ]
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.suffix == ".parquet":
        import ast

        import pandas as pd

        df = pd.read_parquet(p)
        out: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            answers = row.get("answers")
            if isinstance(answers, str):
                try:
                    answers = ast.literal_eval(answers)
                except Exception:
                    answers = [row.get("answer")]
            out.append(
                {
                    "id": str(row.get("id") or ""),
                    "question": str(row.get("question") or ""),
                    "answer": str(row.get("answer") or ""),
                    "answers": answers if isinstance(answers, list) else [str(row.get("answer") or "")],
                    "source": str(row.get("source") or "gsm8k"),
                }
            )
        return out
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError("tasks JSON must be a list")


def _load_batch(path: str):
    batch, _signal = _load_payload(path)
    return batch


def _load_payload(path: str):
    from workflow.contracts import TrajectoryBatch

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "batch" in raw:
        return TrajectoryBatch.model_validate(raw["batch"]), raw.get("train_signal")
    return TrajectoryBatch.model_validate(raw), None


def cmd_collect(args: argparse.Namespace) -> int:
    _ensure_tir_path()
    from workflow import Collector, batch_to_train_signal

    tasks = _load_tasks(args.tasks)
    collector = Collector(
        mock=args.mock,
        endpoint=args.endpoint,
        model=args.model,
        n=args.n,
        spec_path=args.spec,
    )
    batch = collector.collect(tasks)
    signal = batch_to_train_signal(batch, algo=args.algo)
    payload = {
        "batch": batch.model_dump(mode="json"),
        "train_signal": {
            "advantage": signal.advantage.model_dump(mode="json"),
            "loss": signal.loss.model_dump(mode="json"),
            "meta": signal.meta,
        },
    }
    if args.mock and "agentlightning" in sys.modules:
        print("WARNING: agentlightning was imported during --mock collect", file=sys.stderr)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out} n={batch.meta.get('n_trajectories')} mean_reward={batch.meta.get('mean_reward')}")
    else:
        print(json.dumps({"n": batch.meta.get("n_trajectories"), "mean_reward": batch.meta.get("mean_reward")}))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    _ensure_tir_path()
    from workflow.harness import HARNESS

    batch = _load_batch(args.traj)
    hyps = HARNESS.diagnose({"batch": batch, "metrics_path": args.metrics})
    rows = [h.model_dump(mode="json") for h in hyps]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} n={len(rows)}")
    else:
        print(text)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _ensure_tir_path()
    from science_infra.ui.status import render_status_html
    from workflow.harness import HARNESS

    batch, train_signal = _load_payload(args.traj)
    hyps = HARNESS.diagnose({"batch": batch, "metrics_path": args.metrics})
    tb = args.tensorboard or os.environ.get("TENSORBOARD_DIR") or ""
    html = render_status_html(batch, hyps, dashboard=tb, train_signal=train_signal)
    out = Path(args.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    if tb:
        print(f"dashboard={tb}")
    if getattr(args, "open_html", False):
        import webbrowser

        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception as exc:
            print(f"open failed: {exc}", file=sys.stderr)
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    load_science_env()
    env_path = env_loaded_from()
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"dotenv: {env_path if env_path else '(not found)'}")
    base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or ""
    print(f"OPENAI_API_BASE={base or '(unset)'}")
    print(f"OPENAI_MODEL={model or '(unset)'}")
    print(f"OPENAI_API_KEY={'set' if os.environ.get('OPENAI_API_KEY') else '(unset)'}")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output([smi, "-L"], text=True, timeout=5)
            print("nvidia-smi:")
            print(out.strip() or "(empty)")
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
    else:
        print("nvidia-smi: not found")
    try:
        import agentlightning as agl  # noqa: F401

        print(f"agentlightning: yes version={getattr(agl, '__version__', '?')}")
    except Exception as e:
        print(f"agentlightning: no ({e})")
    _ensure_tir_path()
    try:
        from workflow.spec import load_spec

        spec = load_spec()
        print(f"mas_spec topology={spec.topology} tools={spec.tools}")
    except Exception as e:
        print(f"mas_spec: failed ({e})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start Science Control Plane (FastAPI) + optional WebUI static."""
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn/fastapi required: pip install 'fastapi' 'uvicorn[standard]'",
            file=sys.stderr,
        )
        return 1
    host = args.host
    port = int(args.port)
    print(f"Science Control UI → http://{host}:{port}/  (API docs /docs)")
    uvicorn.run(
        "science_infra.control.app:app",
        host=host,
        port=port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="science-infra")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("collect", help="MAS data-layer collect (no VERL)")
    p_c.add_argument("--mock", action="store_true")
    p_c.add_argument("--endpoint", default=None)
    p_c.add_argument("--model", default=None)
    p_c.add_argument("--tasks", default=None)
    p_c.add_argument("--n", type=int, default=1)
    p_c.add_argument("--out", default=None)
    p_c.add_argument("--algo", default="grpo")
    p_c.add_argument("--spec", default=None)
    p_c.set_defaults(func=cmd_collect)

    p_d = sub.add_parser("diagnose", help="Run harness plugins on a traj JSON")
    p_d.add_argument("traj")
    p_d.add_argument("--metrics", default=None)
    p_d.add_argument("--out", default=None)
    p_d.set_defaults(func=cmd_diagnose)

    p_s = sub.add_parser("status", help="Write Science MAS HTML dashboard")
    p_s.add_argument("traj")
    p_s.add_argument("--html", required=True)
    p_s.add_argument("--metrics", default=None)
    p_s.add_argument("--tensorboard", default=None)
    p_s.add_argument("--open", action="store_true", dest="open_html")
    p_s.set_defaults(func=cmd_status)

    p_dash = sub.add_parser("dashboard", help="Alias of status; default HTML path + optional --open")
    p_dash.add_argument("traj")
    p_dash.add_argument("--html", default="science-dashboard.html")
    p_dash.add_argument("--metrics", default=None)
    p_dash.add_argument("--tensorboard", default=None)
    p_dash.add_argument("--open", action="store_true", dest="open_html")
    p_dash.set_defaults(func=cmd_status)

    p_doc = sub.add_parser("doctor", help="Env / GPU / AGL probe")
    p_doc.set_defaults(func=cmd_doctor)

    p_serve = sub.add_parser("serve", help="Science Control UI (FastAPI + WebUI)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
