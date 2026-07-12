"""Stage 3 — annotate.

Human review/correction of prelabels into final taxonomy §7.1 annotations. The
editing logic lives in ``AnnotationSession`` (pure, testable); the Textual TUI
(``tui.annotate_app``) is a thin wrapper over it. The stage enforces the closed
flag vocabulary and the REJECT-must-have-reason / FIX-must-have-fix_actions
rules via the pydantic schema on save.

Modes:
  - interactive (default): launch the TUI.
  - ``auto=True``: non-interactive. If a sidecar (``<name>.json`` next to the
    clip) carries ground-truth labels, apply it; otherwise accept the prelabel.
    Used by ``pipeline run all`` and tests.

Artifacts: ``<workdir>/annotate/annotations.jsonl`` (final §7.1 records).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import PipelineConfig
from ..schema import AnnotationRecord
from ..taxonomy_schema import (
    DIMENSIONS,
    HARD_FAIL_FLAGS,
    VERDICTS,
    duration_bucket,
)
from ..utils.io import read_json, read_jsonl, write_jsonl


def sidecar_to_record(sidecar: dict, prelabel: dict) -> dict:
    """Build a §7.1 record dict from a sample sidecar's expected labels,
    inheriting file_path/duration from the prelabel."""
    verdict = sidecar["expected_verdict"]
    flags = list(sidecar.get("expected_flags", []))
    scores = dict(sidecar.get("expected_scores", {}) or {})
    rec = {
        "asset_id": sidecar.get("asset_id", prelabel["asset_id"]),
        "file_path": prelabel.get("file_path", sidecar.get("file", "")),
        "duration_sec": prelabel.get("duration_sec"),
        "verdict": verdict,
        "hard_fail_flags": flags,
        "scores": scores,
        "context_tags": sidecar.get("context_tags", prelabel.get("context_tags", {})),
        "fix_actions": list(sidecar.get("fix_actions", [])),
        "reject_reason": sidecar.get("reject_reason", ""),
        "needs_human_review": False,
        "reviewer": "sidecar-ground-truth",
        "frame_evidence": [],
    }
    if flags:
        rec["frame_evidence"] = [
            {"frame_time_sec": 0.0, "issue": sidecar.get("note", f), "flag_or_dimension": f}
            for f in flags
        ]
    return rec


def _find_sidecar(file_path: str) -> Optional[dict]:
    p = Path(file_path)
    sc = p.with_suffix(".json")
    if sc.exists():
        try:
            return json.loads(sc.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


class AnnotationSession:
    """In-memory editable set of annotation records with taxonomy guardrails."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.idx = 0
        self.records: list[dict] = self._load()

    # ---- loading --------------------------------------------------------
    def _load(self) -> list[dict]:
        pre_path = self.cfg.stage_dir("prelabel") / "prelabels.jsonl"
        if pre_path.exists():
            return read_jsonl(pre_path)
        # fall back to ingest index (no prelabels yet)
        idx_path = self.cfg.stage_dir("ingest") / "index.json"
        if not idx_path.exists():
            raise FileNotFoundError("run ingest (and ideally prelabel) first")
        index = read_json(idx_path)
        recs = []
        for a in index["assets"]:
            recs.append({
                "asset_id": a["asset_id"], "file_path": a["file_path"],
                "duration_sec": a.get("duration_sec"), "verdict": "PASS",
                "hard_fail_flags": [], "scores": {},
                "context_tags": {"duration_bucket": duration_bucket(a["duration_sec"])
                                 if a.get("duration_sec") else ""},
                "fix_actions": [], "reject_reason": "", "needs_human_review": True,
                "frame_evidence": [], "reviewer": "",
            })
        return recs

    # ---- navigation -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def current(self) -> dict:
        return self.records[self.idx]

    def go_next(self) -> None:
        self.idx = min(self.idx + 1, len(self.records) - 1)

    def go_prev(self) -> None:
        self.idx = max(self.idx - 1, 0)

    def go_to(self, i: int) -> None:
        self.idx = max(0, min(i, len(self.records) - 1))

    # ---- editing --------------------------------------------------------
    def set_verdict(self, verdict: str) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        self.current()["verdict"] = verdict

    def toggle_flag(self, flag: str) -> None:
        if flag not in HARD_FAIL_FLAGS:
            raise ValueError(f"{flag} not in canonical closed set")
        rec = self.current()
        flags = rec.setdefault("hard_fail_flags", [])
        ev = rec.setdefault("frame_evidence", [])
        if flag in flags:
            flags.remove(flag)
            rec["frame_evidence"] = [e for e in ev if e.get("flag_or_dimension") != flag]
        else:
            flags.append(flag)
            ev.append({"frame_time_sec": 0.0, "issue": flag, "flag_or_dimension": flag})
            # a flag forces REJECT and needs a reason (§1/§2)
            rec["verdict"] = "REJECT"
            if not rec.get("reject_reason"):
                rec["reject_reason"] = flag

    def set_score(self, dim: str, value: Optional[int]) -> None:
        if dim not in DIMENSIONS:
            raise ValueError(f"{dim} not a canonical dimension")
        if value is not None and not (0 <= value <= 4):
            raise ValueError("score must be 0-4 or None")
        self.current().setdefault("scores", {})[dim] = value

    def set_reason(self, text: str) -> None:
        self.current()["reject_reason"] = text

    def set_fix_actions(self, actions: list[str]) -> None:
        self.current()["fix_actions"] = list(actions)

    def set_review_flag(self, value: bool) -> None:
        self.current()["needs_human_review"] = value

    # ---- ground-truth / auto -------------------------------------------
    def apply_sidecar(self, i: Optional[int] = None) -> bool:
        i = self.idx if i is None else i
        rec = self.records[i]
        sc = _find_sidecar(rec.get("file_path", ""))
        if sc is None:
            return False
        self.records[i] = sidecar_to_record(sc, rec)
        return True

    def apply_all_sidecars(self) -> int:
        n = 0
        for i in range(len(self.records)):
            if self.apply_sidecar(i):
                n += 1
        return n

    # ---- validation / save ---------------------------------------------
    def validate(self, i: Optional[int] = None) -> tuple[bool, str]:
        rec = self.records[self.idx if i is None else i]
        try:
            AnnotationRecord.model_validate(rec)
            return True, ""
        except Exception as e:  # pydantic ValidationError
            return False, str(e).splitlines()[0] if str(e) else "invalid"

    def validated_records(self) -> tuple[list[dict], list[tuple[str, str]]]:
        ok: list[dict] = []
        errors: list[tuple[str, str]] = []
        for rec in self.records:
            try:
                model = AnnotationRecord.model_validate(rec)
                ok.append(model.model_dump())
            except Exception as e:
                errors.append((rec.get("asset_id", "?"), str(e).splitlines()[0]))
        return ok, errors

    def save(self, reviewer: str = "") -> tuple[Path, int, list[tuple[str, str]]]:
        ok, errors = self.validated_records()
        if reviewer:
            for r in ok:
                if not r.get("reviewer"):
                    r["reviewer"] = reviewer
        out = self.cfg.stage_dir("annotate") / "annotations.jsonl"
        write_jsonl(out, ok)
        return out, len(ok), errors


def run(cfg: PipelineConfig, auto: bool = False, reviewer: str = "") -> dict[str, Any]:
    session = AnnotationSession(cfg)
    if auto:
        n_side = session.apply_all_sidecars()
        out, n_ok, errors = session.save(reviewer=reviewer or "auto")
        counts = _verdict_counts(session.records)
        return {
            "stage": "annotate", "mode": "auto",
            "n_records": len(session), "n_from_sidecar": n_side,
            "n_saved": n_ok, "n_invalid": len(errors),
            "verdicts": counts, "annotations_path": str(out),
            "errors": errors[:5],
        }
    # interactive
    from tui.annotate_app import AnnotateApp

    app = AnnotateApp(session)
    app.run()
    out, n_ok, errors = session.save(reviewer=reviewer or "human")
    return {
        "stage": "annotate", "mode": "interactive",
        "n_records": len(session), "n_saved": n_ok, "n_invalid": len(errors),
        "annotations_path": str(out),
    }


def _verdict_counts(records: list[dict]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for r in records:
        counts[r.get("verdict", "PASS")] = counts.get(r.get("verdict", "PASS"), 0) + 1
    return counts
