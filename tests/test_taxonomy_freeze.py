"""Assert the frozen constants in taxonomy_schema.py still match taxonomy.md.

taxonomy.md is the human source of truth; this test is the mechanical guard
that the code's closed vocabulary has not drifted from the prose (§1/§2/§3/§7).
"""

from __future__ import annotations

import re

from tests.conftest import TAXONOMY_MD
from video_screener import taxonomy_schema as ts


def _md() -> str:
    return TAXONOMY_MD.read_text()


def test_taxonomy_md_exists_and_version():
    text = _md()
    assert f"v{ts.TAXONOMY_VERSION}" in text, (
        f"taxonomy.md does not advertise version {ts.TAXONOMY_VERSION}"
    )


def test_hard_fail_flags_match_section_2():
    """The 9 canonical flag IDs (backtick-wrapped in the §2 table) must match
    HARD_FAIL_FLAGS exactly, in order."""
    text = _md()
    # Isolate §2 (from the "## 2." heading to "## 3.").
    sec2 = re.search(r"##\s*2\.\s.*?(?=\n##\s*3\.)", text, re.S)
    assert sec2, "could not locate §2 in taxonomy.md"
    body = sec2.group(0)
    # Flag IDs appear as `id` at the start of table rows: | `id` | 2.x | ...
    row_ids = re.findall(r"^\|\s*`([a-z_]+)`\s*\|\s*(2\.\d)\s*\|", body, re.M)
    parsed_ids = [rid for rid, _sec in row_ids]
    assert parsed_ids == list(ts.HARD_FAIL_FLAGS), (
        f"§2 flag IDs {parsed_ids} != frozen {ts.HARD_FAIL_FLAGS}"
    )
    # section numbers align
    parsed_sections = {rid: sec for rid, sec in row_ids}
    assert parsed_sections == dict(ts.HARD_FAIL_FLAG_SECTIONS)
    assert len(ts.HARD_FAIL_FLAGS) == 9


def test_dimensions_match_section_7_scores():
    """The 6 dimension keys are synced to §7.1 `scores` (snake_case)."""
    text = _md()
    scores_block = re.search(r'"scores"\s*:\s*\{(.*?)\}', text, re.S)
    assert scores_block, "could not locate §7.1 scores object"
    keys = re.findall(r'"([a-z_]+)"\s*:', scores_block.group(1))
    assert keys == list(ts.DIMENSIONS), (
        f"§7.1 scores keys {keys} != frozen {ts.DIMENSIONS}"
    )
    assert len(ts.DIMENSIONS) == 6


def test_gate_mins_present_in_section_3():
    """Every dimension declares 'Gate min' in §3; baseline is 2."""
    text = _md()
    # Count of "Gate min" occurrences should be >= number of dimensions.
    gate_lines = re.findall(r"\*\*Gate min\*\*:\s*(\d)", text)
    assert len(gate_lines) >= len(ts.DIMENSIONS)
    for d, g in ts.GATE_MIN.items():
        assert g == 2, f"frozen baseline gate for {d} must be 2 per §3"


def test_verdicts_match_section_1():
    text = _md()
    for label in ts.VERDICTS:
        assert f"**{label}**" in text, f"verdict {label} not documented in §1"
    assert ts.VERDICTS == ("PASS", "FIX", "REJECT")


def test_worst_frame_dims_are_video_dims():
    # §5: temporal_stability & motion_quality use worst-frame aggregation.
    assert set(ts.WORST_FRAME_DIMENSIONS) == {"temporal_stability", "motion_quality"}
    assert set(ts.WORST_FRAME_DIMENSIONS) | set(ts.MEAN_FRAME_DIMENSIONS) == set(
        ts.DIMENSIONS
    )
    assert not set(ts.WORST_FRAME_DIMENSIONS) & set(ts.MEAN_FRAME_DIMENSIONS)


def test_context_tag_values_are_closed_lists():
    # §4 fields all present.
    for f in ts.CONTEXT_TAG_LIST_FIELDS + ts.CONTEXT_TAG_SCALAR_FIELDS:
        assert f in ts.CONTEXT_TAG_VALUES
    # A couple of spot checks against §4 prose.
    assert "hero" in ts.CONTEXT_TAG_VALUES["asset_role"]
    assert "photorealistic" in ts.CONTEXT_TAG_VALUES["aesthetic_family"]
    assert set(ts.CONTEXT_TAG_VALUES["duration_bucket"]) == {
        "very_short", "short", "medium", "long"
    }


def test_duration_bucket_mapping():
    assert ts.duration_bucket(1.0) == "very_short"
    assert ts.duration_bucket(2.9) == "very_short"
    assert ts.duration_bucket(3.0) == "short"
    assert ts.duration_bucket(8.0) == "medium"
    assert ts.duration_bucket(20.0) == "long"
    assert ts.duration_bucket(100.0) == "long"
