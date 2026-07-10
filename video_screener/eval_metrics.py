"""Pure metric functions for the evaluate stage (unit-tested independently).

Kept free of torch / IO so the metric math can be verified in isolation:
verdict confusion + per-class P/R/F1, per-dimension ordinal metrics (MAE / exact
/ within-1), per-flag precision/recall/F1, and verdict-vs-flags/dims consistency.
"""

from __future__ import annotations

from typing import Optional

from .aggregate import consistency_violations
from .taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, VERDICTS


def verdict_confusion(y_true: list[str], y_pred: list[str]) -> dict:
    """3x3 confusion matrix (rows=true, cols=pred) + accuracy + per-class PRF."""
    idx = {v: i for i, v in enumerate(VERDICTS)}
    n = len(VERDICTS)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(n))
    per_class = {}
    f1s = []
    for i, v in enumerate(VERDICTS):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i][c] for c in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(matrix[i])
        per_class[v] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {
        "labels": list(VERDICTS),
        "matrix": matrix,
        "accuracy": (correct / total) if total else 0.0,
        "n": total,
        "per_class": per_class,
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
    }


def dimension_metrics(
    true_by_dim: dict[str, list[Optional[int]]],
    pred_by_dim: dict[str, list[Optional[int]]],
) -> dict:
    """Per-dimension MAE / exact-accuracy / within-1-accuracy over non-N/A pairs."""
    out: dict[str, dict] = {}
    for d in DIMENSIONS:
        ts = true_by_dim.get(d, [])
        ps = pred_by_dim.get(d, [])
        errs = []
        exact = 0
        within1 = 0
        n = 0
        for t, p in zip(ts, ps):
            if t is None or p is None:
                continue
            n += 1
            e = abs(int(t) - int(p))
            errs.append(e)
            exact += (e == 0)
            within1 += (e <= 1)
        out[d] = {
            "n": n,
            "mae": (sum(errs) / n) if n else None,
            "exact_acc": (exact / n) if n else None,
            "within1_acc": (within1 / n) if n else None,
        }
    return out


def flag_metrics(
    true_flags: list[list[str]], pred_flags: list[list[str]]
) -> dict:
    """Per-flag precision/recall/F1 + support, plus micro-averaged totals."""
    per_flag: dict[str, dict] = {}
    tot_tp = tot_fp = tot_fn = 0
    for flag in HARD_FAIL_FLAGS:
        tp = fp = fn = 0
        for t, p in zip(true_flags, pred_flags):
            in_t = flag in t
            in_p = flag in p
            tp += in_t and in_p
            fp += (not in_t) and in_p
            fn += in_t and (not in_p)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)
              if (prec and rec) else (0.0 if (tp + fp + fn) else None))
        per_flag[flag] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
        }
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else None
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else None
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p and micro_r) else None)
    return {
        "per_flag": per_flag,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "tp": tot_tp, "fp": tot_fp, "fn": tot_fn},
    }


def stratified_verdict_accuracy(
    strata: list[str], y_true: list[str], y_pred: list[str]
) -> dict:
    """Verdict accuracy grouped by a stratum label (e.g. aesthetic_family)."""
    groups: dict[str, list[bool]] = {}
    for s, t, p in zip(strata, y_true, y_pred):
        groups.setdefault(s or "(none)", []).append(t == p)
    return {
        g: {"n": len(v), "accuracy": (sum(v) / len(v)) if v else 0.0}
        for g, v in groups.items()
    }


def consistency_rate(
    pred_verdicts: list[str],
    pred_scores: list[dict[str, Optional[float]]],
    pred_flags: list[list[str]],
) -> dict:
    """Fraction of predictions whose verdict is INCONSISTENT with the predicted
    flags/dims (§1 coherence): e.g. a PASS co-occurring with a dim=0 or a flag."""
    n = len(pred_verdicts)
    violating = 0
    examples: list[dict] = []
    for i in range(n):
        viols = consistency_violations(pred_verdicts[i], pred_scores[i], pred_flags[i])
        if viols:
            violating += 1
            if len(examples) < 10:
                examples.append({"index": i, "verdict": pred_verdicts[i],
                                 "violations": viols})
    return {
        "n": n,
        "n_inconsistent": violating,
        "rate": (violating / n) if n else 0.0,
        "examples": examples,
    }
