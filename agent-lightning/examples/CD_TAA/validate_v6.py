"""v6 causal-intervention validation: run ONE GAIA sample, capture logs.

Goal: confirm
  (1) single failed step LLM call count <= current impl,
  (2) Level 2a candidate set executes in parallel,
  (3) first COMPLETE candidate stops the intervention.
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Make the MAS package importable regardless of cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from MAS.epc_aw.solver import construct_solver

llm_engine_name = os.getenv("MODEL_Name")
enabled_tools = [
    "Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool",
    "Web_Search_Tool", "Google_Search_Tool", "Screenshot_Tool", "Vision_OCR_Tool",
]
tool_engine = [llm_engine_name] * len(enabled_tools)

DATA = os.path.join(ROOT, "test", "gaia", "data", "data.json")
with open(DATA, "r", encoding="utf-8") as f:
    samples = json.load(f)

# Pick a sample index from argv (default 0)
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sample = samples[idx]
query = sample["question"]
gt = sample.get("answer", "")
print(f"=== v6 validation: pid={sample.get('pid')} idx={idx} ===", flush=True)
print(f"Q: {query[:200]}", flush=True)
print(f"GT: {gt}", flush=True)

solver = construct_solver(
    llm_engine_name=llm_engine_name,
    enabled_tools=enabled_tools,
    tool_engine=tool_engine,
    n=1,
    temperature=0.0,
    max_steps=8,
    verbose=True,
)

t0 = time.time()
try:
    resp = solver.solve(query)
    pred = (resp.get("direct_output") or "").strip()
except Exception as e:
    import traceback
    traceback.print_exc()
    pred = f"<error: {e}>"

dt = time.time() - t0
print(f"\n=== DONE in {dt:.1f}s ===", flush=True)
print(f"PRED: {pred}", flush=True)
