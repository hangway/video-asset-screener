"""ffprobe signalstats/blurdetect backend: JSON parsing, real runs, failure."""

from __future__ import annotations

from video_screener.utils.ffprobe_metrics import (
    SIGNAL_KEYS,
    SignalStats,
    parse_signal_frames,
    probe_signal_stats,
)
from tests.conftest import requires_ffmpeg

FIXTURE = {
    "frames": [
        {
            "pts_time": "0.000000",
            "tags": {
                "lavfi.signalstats.YAVG": "145.866",
                "lavfi.signalstats.YDIF": "0",
                "lavfi.signalstats.YLOW": "93",
                "lavfi.signalstats.YHIGH": "170",
                "lavfi.signalstats.TOUT": "0.000221354",
                "lavfi.blur": "7.592424",
            },
        },
        {   # second frame lacks blurdetect output + has junk TOUT
            "pts_time": "0.041667",
            "tags": {
                "lavfi.signalstats.YAVG": "20.5",
                "lavfi.signalstats.YDIF": "3.25",
                "lavfi.signalstats.YLOW": "10",
                "lavfi.signalstats.YHIGH": "31",
                "lavfi.signalstats.TOUT": "not-a-number",
            },
        },
    ]
}


def test_parse_fixture_json():
    frames = parse_signal_frames(FIXTURE)
    assert len(frames) == 2
    f0, f1 = frames
    assert f0["time_sec"] == 0.0
    assert abs(f0["YAVG"] - 145.866) < 1e-6
    assert f0["YDIF"] == 0.0 and f0["YLOW"] == 93.0 and f0["YHIGH"] == 170.0
    assert abs(f0["TOUT"] - 0.000221354) < 1e-9
    assert abs(f0["blur"] - 7.592424) < 1e-6
    assert abs(f1["time_sec"] - 0.041667) < 1e-9
    assert f1["blur"] is None            # missing tag -> None, not KeyError
    assert f1["TOUT"] is None            # unparseable value -> None
    assert set(SIGNAL_KEYS) <= set(f0)


def test_parse_degenerate_docs():
    assert parse_signal_frames({}) == []
    assert parse_signal_frames({"frames": []}) == []
    frames = parse_signal_frames({"frames": [{"pts_time": None, "tags": None}]})
    assert frames[0]["time_sec"] is None and frames[0]["YAVG"] is None


@requires_ffmpeg
def test_real_run_on_two_samples(samples_dir):
    clean = probe_signal_stats(samples_dir / "clean_pass.mp4")
    dark = probe_signal_stats(samples_dir / "underexposed.mp4")
    for res in (clean, dark):
        assert isinstance(res, SignalStats) and res.ok, res.error
        assert len(res.frames) >= 2
        for fr in res.frames:
            assert set(SIGNAL_KEYS) <= set(fr)
            assert 0.0 <= fr["YAVG"] <= 255.0
        times = [fr["time_sec"] for fr in res.frames]
        assert times == sorted(times)     # frames arrive in time order
    # the underexposed clip must read darker than the clean one
    mean = lambda r: sum(f["YAVG"] for f in r.frames) / len(r.frames)
    assert mean(dark) < mean(clean)


@requires_ffmpeg
def test_corrupted_clip_does_not_crash(samples_dir):
    res = probe_signal_stats(samples_dir / "corrupted.mp4")
    assert res.ok is False
    assert res.frames == []
    assert res.error                      # diagnostic preserved


def test_missing_file_does_not_crash(tmp_path):
    res = probe_signal_stats(tmp_path / "nope.mp4")
    assert res.ok is False and res.frames == []


# ------------- lavfi path escaping (audit A5: Windows drive letters) --------
def test_lavfi_escape_windows_drive_letter_colon():
    """Platform-independent guard for the movie= graph construction.

    ffmpeg unescapes in two levels: the filtergraph parser strips the outer
    quotes (quoted text literal), then the movie option parser splits on ':'
    and unescapes '\\'. An unescaped drive-letter colon terminates the
    filename option at 'C' — the 5 Windows failures reported in PR #10.
    """
    from video_screener.utils.ffprobe_metrics import _escape_lavfi_path

    assert _escape_lavfi_path("C:\\clips\\a.mp4") == "'C\\:\\\\clips\\\\a.mp4'"
    # POSIX paths gain nothing but the quotes
    assert _escape_lavfi_path("/data/clips/a.mp4") == "'/data/clips/a.mp4'"
    # embedded quote handling unchanged
    assert _escape_lavfi_path("/d/o'k.mp4") == "'/d/o\\'k.mp4'"


def test_lavfi_graph_uses_escaped_path():
    from video_screener.utils.ffprobe_metrics import _escape_lavfi_path

    escaped = _escape_lavfi_path("C:\\clips\\a.mp4")
    graph = f"movie={escaped},signalstats"
    assert graph == "movie='C\\:\\\\clips\\\\a.mp4',signalstats"
