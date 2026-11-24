#!/bin/bash
# Sequential val-only for the checkpoints still on disk.
set -e
cd /root/autodl-tmp/MAS_Science_Infra
chmod +x experiments/fcr-step4-retrain/launch_valonly.sh
MD=experiments/fcr-step4-retrain/val_ra_rc.md
for STEP in 5 11 12 13; do
  LOG="experiments/fcr-step4-retrain/valonly_step${STEP}.log"
  echo "=== val step ${STEP} $(date -Is) ===" | tee -a "$LOG"
  bash experiments/fcr-step4-retrain/launch_valonly.sh "$STEP" >> "$LOG" 2>&1 || echo "VAL_FAIL step ${STEP}" | tee -a "$LOG"
  python3 - <<PY
import re
from pathlib import Path
log = Path("experiments/fcr-step4-retrain/valonly_step${STEP}.log").read_text(errors="ignore")
def grab(key):
    ms = re.findall(rf"{re.escape(key)}:([-\d.eE+]+)", log)
    return ms[-1] if ms else ""
ra, rc = grab("val/fcr/R_answer_mean"), grab("val/fcr/R_correct_mean")
status = "ok" if ra else "missing"
md = Path("experiments/fcr-step4-retrain/val_ra_rc.md")
text = md.read_text()
old = f"| ${STEP} | | | pending |"
new = f"| ${STEP} | {ra or '—'} | {rc or '—'} | {status} |"
if old not in text:
    # already filled or spacing differs
    text = re.sub(rf"\| ${STEP} \|.*?\|.*?\|.*?\|", new, text, count=1)
else:
    text = text.replace(old, new, 1)
md.write_text(text)
print(f"step ${STEP} val_Ra={ra} val_Rc={rc} status={status}")
PY
done
echo ALL_VAL_DONE
