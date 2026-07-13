"""HTML-injection escaping in the screen report and dashboard (audit A8).

Asset ids come from filenames on disk — attacker-influenceable in any
shared-intake setting — and were interpolated into report/dashboard markup
unescaped.
"""

from __future__ import annotations

from video_screener.stages.screen import _build_report

PAYLOAD = "<img src=x onerror=1>"


def _routing():
    return {"n": 1, "n_needs_review": 0,
            "counts": {"PASS": 1, "FIX": 0, "REJECT": 0}}


def _row(asset_id: str) -> dict:
    return {
        "asset_id": asset_id, "verdict": "PASS", "confidence": 1.0,
        "hard_fail_flags": [], "scores": {}, "fix_actions": [],
        "primary_reasons": [], "needs_human_review": False, "thumb": "",
    }


def test_report_card_escapes_asset_id():
    html_out = _build_report([_row(f"clip_{PAYLOAD}")], _routing())
    assert PAYLOAD not in html_out
    assert "clip_&lt;img src=x onerror=1&gt;" in html_out


def test_report_consistency_section_escapes_asset_and_subject():
    cons_doc = {
        "threshold": 0.5, "n_below_threshold": 0, "n_drift_flagged": 0,
        "clips": [{
            "asset_id": f"clip_{PAYLOAD}", "subject": f"subj_{PAYLOAD}",
            "score": 0.9, "max_drift": 0.1, "below_threshold": False,
            "edge_stability": None,
        }],
    }
    html_out = _build_report([], _routing(), cons_doc, None)
    assert PAYLOAD not in html_out
    assert "clip_&lt;img src=x onerror=1&gt;" in html_out
    assert "subj_&lt;img src=x onerror=1&gt;" in html_out


def test_dashboard_payload_cannot_break_out_of_script(tmp_path):
    from dashboard.build import build_dashboard
    from video_screener.utils.io import write_json

    evil = f"clip</script>{PAYLOAD}"
    index = {
        "n_assets": 1, "n_duplicates": 0, "taxonomy_version": "0.3.1",
        "assets": [{"asset_id": evil, "sampled_frames": [],
                    "duration_sec": 8.0, "duplicate_of": None}],
    }
    write_json(tmp_path / "ingest" / "index.json", index)
    html_out = build_dashboard(tmp_path).read_text(encoding="utf-8")
    # every '<' in the JSON payload is <-escaped, so neither the
    # </script> terminator nor the img tag can appear as markup
    assert "</script>" + PAYLOAD not in html_out
    assert PAYLOAD not in html_out
    assert "\\u003cimg src=x onerror=1>" in html_out
