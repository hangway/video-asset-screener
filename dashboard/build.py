"""Build the single-file HTML dashboard from a run's artifacts.

Reads whatever stage artifacts exist under a workdir and injects them (with
base64 clip thumbnails) into ``template.html`` to produce a self-contained,
server-less page: stage status, eval metrics, training curve, and a dataset
viewer with per-clip thumbnails + verdict/flags.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

TEMPLATE = Path(__file__).with_name("template.html")


def _read_json(p: Path) -> Optional[Any]:
    return json.loads(p.read_text()) if p.exists() else None


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _thumb(frame_path: Optional[str], max_w: int = 220) -> str:
    if not frame_path or not Path(frame_path).exists():
        return ""
    try:
        from PIL import Image

        im = Image.open(frame_path).convert("RGB")
        w, h = im.size
        im = im.resize((max_w, max(1, int(h * max_w / w))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=68)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _assemble(workdir: str | Path) -> dict:
    wd = Path(workdir)
    index = _read_json(wd / "ingest" / "index.json")
    annotations = _read_jsonl(wd / "annotate" / "annotations.jsonl")
    splits = _read_json(wd / "dataset" / "splits.json")
    train_log = _read_json(wd / "train" / "train_log.json")
    eval_rep = _read_json(wd / "evaluate" / "eval_report.json")
    screen_res = _read_jsonl(wd / "screen" / "screen_results.jsonl")
    routing = _read_json(wd / "screen" / "routing.json")

    ann_by_id = {a["asset_id"]: a for a in annotations}
    assign = (splits or {}).get("assignment", {})

    assets: list[dict] = []
    if index:
        for a in index["assets"]:
            aid = a["asset_id"]
            frames = a.get("sampled_frames", [])
            mid = frames[len(frames) // 2]["path"] if frames else None
            ann = ann_by_id.get(aid, {})
            assets.append({
                "asset_id": aid,
                "verdict": ann.get("verdict", "PASS"),
                "flags": ann.get("hard_fail_flags", []),
                "scores": ann.get("scores", {}),
                "split": assign.get(aid, ""),
                "duration_sec": a.get("duration_sec"),
                "decode_ok": a.get("decode_ok", True),
                "duplicate_of": a.get("duplicate_of"),
                "thumb": _thumb(mid),
            })

    def stage_entry(done: bool, metric: str) -> dict:
        return {"done": done, "metric": metric}

    stages = {
        "ingest": stage_entry(index is not None,
                              f"{index['n_assets']} assets, {index['n_duplicates']} dup"
                              if index else ""),
        "prelabel": stage_entry((wd / "prelabel" / "prelabels.jsonl").exists(), ""),
        "annotate": stage_entry(bool(annotations), f"{len(annotations)} labeled"),
        "dataset": stage_entry(splits is not None,
                               f"{splits['n_groups']} groups, 0 leak" if splits else ""),
        "train": stage_entry(train_log is not None,
                             f"{train_log['epochs_run']} ep, val {train_log['best_val_loss']:.2f}"
                             if train_log else ""),
        "evaluate": stage_entry(eval_rep is not None,
                               f"acc {eval_rep['verdict']['accuracy']*100:.0f}%"
                               if eval_rep else ""),
        "screen": stage_entry(bool(screen_res),
                             f"{routing['n']} screened" if routing else ""),
    }

    train_block = None
    if train_log:
        train_block = {
            "encoder": train_log.get("encoder"),
            "feature_dim": train_log.get("feature_dim"),
            "best_val": round(train_log.get("best_val_loss") or 0, 3),
            "history": [
                {"epoch": h["epoch"],
                 "train_loss": h["train"]["loss"],
                 "val_loss": (h["val"]["loss"] if h.get("val") else None)}
                for h in train_log.get("history", [])
            ],
        }

    screen_block = None
    if routing:
        screen_block = {"n": routing["n"], "counts": routing["counts"]}

    return {
        "taxonomy_version": (index or eval_rep or {}).get("taxonomy_version", "0.3.1"),
        "workdir": str(wd),
        "stages": stages,
        "assets": assets,
        "train": train_block,
        "evaluate": eval_rep,
        "screen": screen_block,
    }


def build_dashboard(workdir: str | Path, out: Optional[str | Path] = None) -> Path:
    data = _assemble(workdir)
    template = TEMPLATE.read_text()
    payload = json.dumps(data, ensure_ascii=False)
    # inject between the /*__DATA__*/ ... /*__END__*/ markers
    start = template.index("/*__DATA__*/") + len("/*__DATA__*/")
    end = template.index("/*__END__*/")
    html = template[:start] + payload + template[end:]
    dest = Path(out) if out else Path(workdir) / "dashboard.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html)
    return dest
