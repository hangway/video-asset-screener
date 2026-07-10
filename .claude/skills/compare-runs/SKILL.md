---
name: compare-runs
description: Compare two video-asset-screener runs — verdict accuracy/F1, per-dimension MAE, per-flag F1, consistency rate, and screen routing distribution. Use when the user wants to compare experiments, check whether a config change helped, or diff two run workdirs.
---

# compare-runs

Diff the eval + screen artifacts of two runs to see what changed.

## Inputs (per run)
- `runs/<name>/evaluate/eval_report.json` — verdict/dim/flag metrics + consistency
- `runs/<name>/screen/routing.json`       — PASS/FIX/REJECT counts + needs_review
- `runs/<name>/train/train_log.json`      — loss curve, encoder, epochs

## How to compare
```bash
python - <<'PY'
import json
from pathlib import Path
A, B = "runs/runA", "runs/runB"     # adjust to the two workdirs
def load(p):
    p = Path(p)
    return (json.loads((p/"evaluate/eval_report.json").read_text()) if (p/"evaluate/eval_report.json").exists() else None,
            json.loads((p/"screen/routing.json").read_text()) if (p/"screen/routing.json").exists() else None,
            json.loads((p/"train/train_log.json").read_text()) if (p/"train/train_log.json").exists() else None)
ea, ra, ta = load(A); eb, rb, tb = load(B)
def row(name, va, vb):
    print(f"  {name:26s} {va!s:>10s}  ->  {vb!s:>10s}")
print(f"metric{'':22s} {A:>10s}      {B:>10s}")
if ea and eb:
    row("verdict accuracy", round(ea['verdict']['accuracy'],3), round(eb['verdict']['accuracy'],3))
    row("verdict macro-F1", round(ea['verdict']['macro_f1'],3), round(eb['verdict']['macro_f1'],3))
    row("flag micro-F1", ea['flags']['micro']['f1'], eb['flags']['micro']['f1'])
    row("consistency (incons.)", round(ea['consistency']['rate'],3), round(eb['consistency']['rate'],3))
    for d in ea['dimensions']:
        row(f"MAE {d}", ea['dimensions'][d]['mae'], eb['dimensions'][d]['mae'])
if ra and rb: row("screen routing", ra['counts'], rb['counts'])
if ta and tb: row("best_val_loss", round(ta['best_val_loss'],3), round(tb['best_val_loss'],3))
PY
```

## What to report
- Direction of each metric (better/worse) and by how much.
- Whether the consistency (inconsistent) rate went up (regression) or down.
- Routing shifts (e.g. more clips sent to REJECT/FIX) and needs_review counts.
- Note when runs used different encoders / epochs (not apples-to-apples).
