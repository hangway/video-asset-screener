---
name: inspect-dataset
description: Inspect a built video-asset-screener dataset — split assignments, leakage-check status, per-verdict/per-flag class balance, and per-clip labels. Use when the user asks about dataset composition, split integrity, class imbalance, or which clips landed in train/val/test.
---

# inspect-dataset

Summarize the artifacts the `dataset` stage produced under a workdir.

## Inputs
- `runs/<name>/dataset/splits.json`  — assignment, groups, group_split, leakage_pairs
- `runs/<name>/dataset/balance_report.json` — per-split verdict/flag/score histograms
- `runs/<name>/dataset/{train,val,test}.jsonl` — training records

## How to inspect
Run this and report the highlights (leakage MUST be empty; call out imbalance):

```bash
python - <<'PY'
import json
from pathlib import Path
wd = Path("runs/default/dataset")   # adjust workdir as needed
splits = json.loads((wd/"splits.json").read_text())
bal = json.loads((wd/"balance_report.json").read_text())
print("groups:", splits["n_groups"], "| leakage pairs:", len(splits["leakage_pairs"]))
assert not splits["leakage_pairs"], "LEAKAGE DETECTED"
for split in ("train","val","test","overall"):
    e = bal["overall"] if split=="overall" else bal["per_split"][split]
    flags = {k:v for k,v in e["flags"].items() if v}
    print(f"{split:7s} n={e['n']:3d} verdicts={e['verdicts']} flags={flags or '{}'}")
print("\nassignment:")
for aid, s in sorted(splits["assignment"].items()):
    print(f"  {aid:16s} -> {s}")
PY
```

## What to flag to the user
- Any non-empty `leakage_pairs` (a bug — near-dups/same-source crossed splits).
- Verdict or flag classes absent from `train` (model can't learn them).
- Severe imbalance (e.g. a flag with support 0 in train but >0 in test).
- Groups that force uneven split sizes (small datasets).
