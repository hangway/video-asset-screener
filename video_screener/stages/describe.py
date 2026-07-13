"""Optional semantic description stage.

The stage invokes the separately installed ``video-analyzer`` console script
for each configured clip.  It stores the tool's JSON output under
``<workdir>/describe/<asset_id>/`` and a stable JSONL manifest at
``<workdir>/describe/analysis.jsonl``.  This is enrichment only: it never
changes the screen records or their PASS/FIX/REJECT routing.

The external package is intentionally not a dependency of the Screener.  A
default run therefore remains offline and deterministic; users opt in by
setting ``describe.enabled: true`` and installing/configuring video-analyzer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..taxonomy_schema import TAXONOMY_VERSION
from ..utils import video
from ..utils.io import read_json, read_jsonl, write_json, write_jsonl


def _asset_id_for(path: Path, taken: set[str]) -> str:
    """Match ingest's stable stem-based IDs when no ingest index exists."""

    stem = path.stem
    aid = stem
    i = 1
    while aid in taken:
        aid = f"{stem}_{i}"
        i += 1
    taken.add(aid)
    return aid


def _assets(cfg: PipelineConfig) -> list[dict[str, str]]:
    """Return asset IDs and paths, preferring the existing ingest contract."""

    index_path = cfg.stage_dir("ingest") / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        assets = [
            {"asset_id": str(a["asset_id"]), "file_path": str(a["file_path"])}
            for a in index.get("assets", [])
        ]
    else:
        taken: set[str] = set()
        assets = [
            {"asset_id": _asset_id_for(path, taken), "file_path": str(path)}
            for path in video.find_videos(cfg.video_dirs)
        ]

    verdict_filter = cfg.describe.screen_verdicts
    if not verdict_filter:
        return assets
    results_path = cfg.stage_dir("screen") / "screen_results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(
            "describe.screen_verdicts requires screen/screen_results.jsonl; "
            "run the screen stage first"
        )
    verdicts = {
        str(row.get("asset_id")): row.get("verdict")
        for row in read_jsonl(results_path)
    }
    return [a for a in assets if verdicts.get(a["asset_id"]) in verdict_filter]


def _command(cfg: PipelineConfig, clip: str, out_dir: Path) -> list[str]:
    """Build a video-analyzer CLI invocation without persisting secrets."""

    dc = cfg.describe
    cmd = [dc.executable, clip, "--output", str(out_dir), "--client", dc.client]
    if dc.ollama_url:
        cmd.extend(["--ollama-url", dc.ollama_url])
    if dc.api_url:
        cmd.extend(["--api-url", dc.api_url])
    if dc.client == "openai_api" and dc.api_key_env:
        api_key = os.environ.get(dc.api_key_env)
        if not api_key:
            raise ValueError(
                f"describe.client=openai_api requires environment variable "
                f"{dc.api_key_env!r} (configured by describe.api_key_env)"
            )
        cmd.extend(["--api-key", api_key])
    if dc.model:
        cmd.extend(["--model", dc.model])
    if dc.prompt:
        cmd.extend(["--prompt", dc.prompt])
    if dc.duration_sec is not None:
        cmd.extend(["--duration", str(dc.duration_sec)])
    if dc.max_frames is not None:
        cmd.extend(["--max-frames", str(dc.max_frames)])
    if dc.whisper_model:
        cmd.extend(["--whisper-model", dc.whisper_model])
    if dc.language:
        cmd.extend(["--language", dc.language])
    if dc.device:
        cmd.extend(["--device", dc.device])
    if dc.temperature is not None:
        cmd.extend(["--temperature", str(dc.temperature)])
    if dc.keep_frames:
        cmd.append("--keep-frames")
    return cmd


def _analysis_path(out_dir: Path) -> Path | None:
    direct = out_dir / "analysis.json"
    if direct.exists():
        return direct
    matches = sorted(out_dir.rglob("analysis.json"))
    return matches[0] if len(matches) == 1 else None


def _error_record(asset_id: str, file_path: str, error: str, **extra: Any) -> dict[str, Any]:
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "stage": "describe",
        "asset_id": asset_id,
        "file_path": file_path,
        "status": "error",
        "error": error,
        **extra,
    }


def _redact_secret(text: str, cfg: PipelineConfig) -> str:
    """Keep provider secrets out of persisted stderr/error artifacts."""

    env_name = cfg.describe.api_key_env
    secret = os.environ.get(env_name) if env_name else None
    return text.replace(secret, "[REDACTED]") if secret else text


def run(cfg: PipelineConfig) -> dict[str, Any]:
    """Run semantic enrichment when explicitly enabled; otherwise no-op."""

    dc = cfg.describe
    if not dc.enabled:
        return {"stage": "describe", "enabled": False, "skipped": True, "n_described": 0}

    executable = shutil.which(dc.executable)
    if executable is None and not Path(dc.executable).exists():
        raise FileNotFoundError(
            f"describe executable {dc.executable!r} was not found on PATH; "
            "install video-analyzer or set describe.executable"
        )

    out_root = cfg.stage_dir("describe")
    out_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    assets = _assets(cfg)

    for asset in assets:
        aid = asset["asset_id"]
        clip = asset["file_path"]
        clip_out = out_root / aid
        clip_out.mkdir(parents=True, exist_ok=True)
        try:
            cmd = _command(cfg, clip, clip_out)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=dc.timeout_sec,
            )
            if result.returncode != 0:
                records.append(_error_record(
                    aid,
                    clip,
                    _redact_secret(result.stderr.strip()[-2000:], cfg)
                    or "video-analyzer failed",
                    returncode=result.returncode,
                ))
                continue
            analysis_path = _analysis_path(clip_out)
            if analysis_path is None:
                records.append(_error_record(
                    aid, clip, "video-analyzer completed without analysis.json"
                ))
                continue
            try:
                analysis = read_json(analysis_path)
            except (OSError, json.JSONDecodeError) as exc:
                records.append(_error_record(
                    aid, clip, f"analysis.json is not valid JSON: {exc}",
                    analysis_path=str(analysis_path),
                ))
                continue
            records.append({
                "taxonomy_version": TAXONOMY_VERSION,
                "stage": "describe",
                "asset_id": aid,
                "file_path": clip,
                "status": "ok",
                "analysis_path": str(analysis_path),
                "client": dc.client,
                "model": dc.model or "",
                "analysis": analysis,
            })
        except subprocess.TimeoutExpired:
            records.append(_error_record(
                aid, clip, f"video-analyzer timed out after {dc.timeout_sec}s"
            ))
        except Exception as exc:
            # One bad clip must not discard useful descriptions for the rest of
            # the batch, but the error remains explicit in analysis.jsonl.
            records.append(_error_record(aid, clip, str(exc)))

    manifest = write_jsonl(out_root / "analysis.jsonl", records)
    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "stage": "describe",
        "enabled": True,
        "client": dc.client,
        "model": dc.model or "",
        "screen_verdicts": dc.screen_verdicts or [],
        "n_assets": len(records),
        "n_described": sum(r.get("status") == "ok" for r in records),
        "n_failed": sum(r.get("status") == "error" for r in records),
        "results_path": str(manifest),
        "verdicts_unchanged": True,
    }
    write_json(out_root / "summary.json", summary)
    return summary
