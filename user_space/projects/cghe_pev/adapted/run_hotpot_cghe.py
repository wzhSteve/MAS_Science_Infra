"""Backward-compatible wrapper around run_cghe_eval for HotpotQA. """

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("CGHE_DATA", str(ROOT / "data" / "hotpotqa.json"))
os.environ.setdefault("CGHE_RUN_TAG", os.environ.get("CGHE_RUN_TAG") or "hotpot")

from run_cghe_eval import main

if __name__ == "__main__":
    main()
