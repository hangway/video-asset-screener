"""Non-ASCII round-trip regression tests for every text write path.

The screen report (✂ trim marker), consistency report, and dashboard all
carry non-ASCII characters. On Windows the default text encoding is the
ANSI code page (cp1252), so any write_text/open("w") without an explicit
encoding crashes with UnicodeEncodeError — which is exactly how this
shipped: the Linux-only suite never saw a non-UTF-8 default. Every
assertion here decodes with an explicit "utf-8" so the tests themselves
never depend on the platform default either.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from video_screener.utils.io import read_json, read_jsonl, write_json, write_jsonl

NON_ASCII = "✂ trim_head 1.0s ⚑ inconsistent ⚠ drift — 🎬"


def test_write_json_round_trips_non_ascii(tmp_path):
    p = write_json(tmp_path / "consistency_report.json", {"marker": NON_ASCII})
    raw = p.read_bytes().decode("utf-8")
    assert NON_ASCII in raw  # ensure_ascii=False kept the literal characters
    assert read_json(p)["marker"] == NON_ASCII


def test_write_jsonl_round_trips_non_ascii(tmp_path):
    rows = [{"asset_id": "clip_✂_01", "note": NON_ASCII}]
    p = write_jsonl(tmp_path / "screen_results.jsonl", rows)
    raw = p.read_bytes().decode("utf-8")
    assert NON_ASCII in raw
    assert read_jsonl(p) == rows


def test_dashboard_html_round_trips_non_ascii(tmp_path):
    from dashboard.build import build_dashboard

    index = {
        "n_assets": 1,
        "n_duplicates": 0,
        "taxonomy_version": "0.3.1",
        "assets": [{"asset_id": "clip_✂_01", "sampled_frames": [], "duration_sec": 8.0}],
    }
    write_json(tmp_path / "ingest" / "index.json", index)
    dest = build_dashboard(tmp_path)
    html = dest.read_bytes().decode("utf-8")
    assert "clip_✂_01" in html


def test_screen_report_round_trips_trim_marker(tmp_path):
    """The real report builder emits the ✂ trim marker; the write must be utf-8."""
    from video_screener.stages.screen import _build_report

    routing = {"n": 0, "n_needs_review": 0, "counts": {"PASS": 0, "FIX": 0, "REJECT": 0}}
    cons_doc = {
        "threshold": 0.5,
        "n_below_threshold": 0,
        "n_drift_flagged": 0,
        "clips": [{
            "asset_id": "clip_01", "subject": "subject_a", "score": 0.9,
            "max_drift": 0.1, "below_threshold": False,
            "edge_stability": {"trim_suggestions": [
                {"action": "trim_head", "suggested_trim_sec": 1.0},
            ]},
        }],
    }
    html = _build_report([], routing, cons_doc, None)
    assert "✂ trim_head 1.0s" in html
    dest = tmp_path / "screen_report.html"
    dest.write_text(html, encoding="utf-8")  # same call shape as screen.run()
    assert "✂ trim_head 1.0s" in dest.read_bytes().decode("utf-8", errors="strict")


# --------------- regression-class guards (Windows default encoding) ---------

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ("video_screener", "dashboard", "tui", "samples")

# Attribute owners whose .open() is not text file IO.
_NON_FILE_OPEN_OWNERS = {"Image", "webbrowser"}


def _text_io_violations(path: Path) -> list[str]:
    """Return 'file:line call' for every text-mode IO call missing encoding=."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if owner in _NON_FILE_OPEN_OWNERS:
                continue
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name not in {"write_text", "read_text", "open"}:
            continue
        kwargs = {k.arg for k in node.keywords}
        if "encoding" in kwargs:
            continue
        if name == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if isinstance(mode, str) and "b" in mode:
                continue  # binary mode: encoding is meaningless
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        bad.append(f"{rel}:{node.lineno} {name}(...)")
    return bad


def test_no_text_io_without_explicit_encoding():
    """Static guard: Item 1's bug class can never ship again.

    Every write_text/read_text/open text-mode call in production code must
    pass encoding= explicitly, because the default is the locale encoding
    (cp1252 on Windows) and reports/artifacts routinely contain non-ASCII.
    """
    violations: list[str] = []
    for d in PRODUCTION_DIRS:
        for py in sorted((REPO_ROOT / d).rglob("*.py")):
            violations += _text_io_violations(py)
    assert not violations, (
        "text IO without explicit encoding= (crashes on Windows/cp1252):\n"
        + "\n".join(violations)
    )


def test_static_guard_actually_detects_violations(tmp_path):
    # prove the guard has teeth: an unencoded write_text must be flagged...
    p = tmp_path / "bad.py"
    p.write_text(
        "from pathlib import Path\n"
        "Path('x').write_text('✂')\n"
        "open('y', 'w').write('✂')\n"
        "Path('z').read_text()\n",
        encoding="utf-8",
    )
    assert len(_text_io_violations(p)) == 3
    # ...while encoded / binary / non-file calls pass clean
    q = tmp_path / "good.py"
    q.write_text(
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "Path('x').write_text('✂', encoding='utf-8')\n"
        "open('y', 'wb').write(b'0')\n"
        "Image.open('f.png')\n",
        encoding="utf-8",
    )
    assert _text_io_violations(q) == []


_ROUNDTRIP_SCRIPT = """
import json
from video_screener.utils.io import read_json, read_jsonl, write_json, write_jsonl
from dashboard.build import build_dashboard
from pathlib import Path
import sys

wd = Path(sys.argv[1])
marker = "\\u2702 trim_head 1.0s \\u2691 \\u26a0 \\U0001f3ac"

p = write_json(wd / "consistency_report.json", {"marker": marker})
assert read_json(p)["marker"] == marker
p = write_jsonl(wd / "screen_results.jsonl", [{"note": marker}])
assert read_jsonl(p)[0]["note"] == marker
write_json(wd / "ingest" / "index.json", {
    "n_assets": 1, "n_duplicates": 0, "taxonomy_version": "0.3.1",
    "assets": [{"asset_id": "clip_\\u2702_01", "sampled_frames": [], "duration_sec": 8.0}],
})
html = build_dashboard(wd).read_bytes().decode("utf-8")
assert "clip_\\u2702_01" in html
print("OK")
"""


def test_write_paths_survive_non_utf8_locale(tmp_path):
    """Runtime guard: round-trip every write path under an ASCII default.

    Re-runs the Item 1 round-trips in a subprocess with LC_ALL=C and UTF-8
    mode disabled, making the platform default encoding ASCII — a stricter
    stand-in for Windows cp1252. Any write path that regresses to the
    platform default raises UnicodeEncodeError here, on Linux CI.
    """
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0")
    env.pop("PYTHONIOENCODING", None)
    res = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", _ROUNDTRIP_SCRIPT, str(tmp_path)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "OK" in res.stdout


def test_json_payloads_keep_non_ascii_unescaped(tmp_path):
    # ensure_ascii=False is the contract: reports stay human-readable, so
    # the utf-8 write encoding is load-bearing, not cosmetic.
    p = write_json(tmp_path / "r.json", {"m": NON_ASCII})
    assert "\\u" not in json.dumps(json.loads(p.read_bytes().decode("utf-8"))["m"], ensure_ascii=False)
