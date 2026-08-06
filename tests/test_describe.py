"""Optional semantic-enrichment stage tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from video_screener.config import PipelineConfig
from video_screener.stages import describe
from video_screener.utils.io import read_jsonl, write_json, write_jsonl


def test_describe_disabled_is_a_noop(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[])

    summary = describe.run(cfg)

    assert summary == {
        "stage": "describe",
        "enabled": False,
        "skipped": True,
        "n_described": 0,
    }
    assert not cfg.stage_dir("describe").exists()


def test_describe_invokes_external_cli_and_writes_manifest(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video; the external tool is mocked")
    cfg = PipelineConfig(
        workdir=str(tmp_path / "run"),
        video_dirs=[str(tmp_path)],
        describe={
            "enabled": True,
            "executable": "video-analyzer",
            "client": "ollama",
            "model": "llama3.2-vision",
            "prompt": "Describe the clip",
            "max_frames": 5,
        },
    )
    calls: list[list[str]] = []

    monkeypatch.setattr(describe.shutil, "which", lambda _: "video-analyzer")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_dir = tmp_path / "run" / "describe" / "clip"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text(json.dumps({
            "transcript": {"text": "hello"},
            "video_description": {"response": "A test clip."},
        }), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(describe.subprocess, "run", fake_run)

    summary = describe.run(cfg)
    rows = read_jsonl(cfg.stage_dir("describe") / "analysis.jsonl")

    assert summary["n_described"] == 1
    assert summary["n_failed"] == 0
    assert rows[0]["status"] == "ok"
    assert rows[0]["analysis"]["video_description"]["response"] == "A test clip."
    assert "--output" in calls[0] and "--max-frames" in calls[0]
    assert (cfg.stage_dir("describe") / "summary.json").exists()


def test_describe_openai_key_is_read_from_environment_not_artifacts(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mock")
    cfg = PipelineConfig(
        workdir=str(tmp_path / "run"),
        video_dirs=[str(tmp_path)],
        describe={
            "enabled": True,
            "client": "openai_api",
            "api_key_env": "VIDEO_ANALYZER_TEST_KEY",
        },
    )
    secret = "test-secret-not-persisted"
    monkeypatch.setenv("VIDEO_ANALYZER_TEST_KEY", secret)
    monkeypatch.setattr(describe.shutil, "which", lambda _: "video-analyzer")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_dir = tmp_path / "run" / "describe" / "clip"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(describe.subprocess, "run", fake_run)
    describe.run(cfg)

    manifest = (cfg.stage_dir("describe") / "analysis.jsonl").read_text(encoding="utf-8")
    assert secret in calls[0]
    assert secret not in manifest


def test_describe_missing_api_key_is_recorded_per_asset(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mock")
    cfg = PipelineConfig(
        workdir=str(tmp_path / "run"),
        video_dirs=[str(tmp_path)],
        describe={
            "enabled": True,
            "client": "openai_api",
            "api_key_env": "VIDEO_ANALYZER_MISSING_KEY",
        },
    )
    monkeypatch.delenv("VIDEO_ANALYZER_MISSING_KEY", raising=False)
    monkeypatch.setattr(describe.shutil, "which", lambda _: "video-analyzer")
    monkeypatch.setattr(
        describe.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLI must not run without the configured API key")
        ),
    )

    summary = describe.run(cfg)
    rows = read_jsonl(cfg.stage_dir("describe") / "analysis.jsonl")

    assert summary["n_described"] == 0
    assert summary["n_failed"] == 1
    assert "VIDEO_ANALYZER_MISSING_KEY" in rows[0]["error"]


def test_describe_can_filter_to_existing_screen_verdicts(tmp_path, monkeypatch):
    pass_clip = tmp_path / "pass_clip.mp4"
    reject_clip = tmp_path / "reject_clip.mp4"
    pass_clip.write_bytes(b"mock")
    reject_clip.write_bytes(b"mock")
    cfg = PipelineConfig(
        workdir=str(tmp_path / "run"),
        video_dirs=[str(tmp_path)],
        describe={"enabled": True, "screen_verdicts": ["PASS"]},
    )
    write_jsonl(cfg.stage_dir("screen") / "screen_results.jsonl", [
        {"asset_id": "pass_clip", "verdict": "PASS"},
        {"asset_id": "reject_clip", "verdict": "REJECT"},
    ])
    monkeypatch.setattr(describe.shutil, "which", lambda _: "video-analyzer")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_dir = tmp_path / "run" / "describe" / Path(cmd[1]).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(describe.subprocess, "run", fake_run)
    summary = describe.run(cfg)

    assert summary["n_assets"] == 1
    assert Path(calls[0][1]).stem == "pass_clip"


def test_dashboard_displays_describe_text_escaped_without_changing_verdict(tmp_path):
    wd = tmp_path / "run"
    write_json(wd / "ingest" / "index.json", {
        "taxonomy_version": "0.3.1",
        "n_assets": 1,
        "n_duplicates": 0,
        "assets": [{
            "asset_id": "clip",
            "duration_sec": 1.0,
            "decode_ok": True,
            "sampled_frames": [],
        }],
    })
    write_jsonl(wd / "describe" / "analysis.jsonl", [{
        "asset_id": "clip",
        "status": "ok",
        "analysis": {
            "video_description": {"response": "<b>unsafe</b>"},
            "transcript": {"text": "say 'hello'"},
        },
    }])
    write_json(wd / "describe" / "summary.json", {
        "n_described": 1, "n_failed": 0, "stage": "describe",
    })

    from dashboard.build import build_dashboard

    html = build_dashboard(wd).read_text(encoding="utf-8")
    # The raw dashboard payload neutralizes '<' before the browser's escaped
    # template rendering runs (the same XSS boundary as existing asset IDs).
    assert r"\u003cb>unsafe\u003c/b>" in html
    assert "<b>unsafe</b>" not in html
    assert "PASS" in html  # enrichment did not create or alter a verdict
