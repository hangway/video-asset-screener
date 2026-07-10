# notes.md — decisions, lessons, open questions (one per line)

## Environment
- No ffmpeg preinstalled; `apt-get install -y ffmpeg` (after one retry) works. torch installs from PyPI (default index) — pytorch.org wheel index is proxy-blocked (403).
- HuggingFace (huggingface.co) is NOT reachable from this sandbox (curl -> 000). Cannot download CLIP/SigLIP weights online.

## Design decisions
- DECISION: encoder is pluggable. `encoder: auto` prefers a real CLIP/SigLIP backbone iff weights are locally available; otherwise falls back to a DETERMINISTIC offline per-frame feature extractor (fixed random projection of multi-scale frame statistics). Keeps the pipeline offline-runnable + tests deterministic while preserving the frozen-features -> temporal-transformer -> 3-heads architecture. Documented in README.
- DECISION: taxonomy_schema.py freezes the closed vocabulary (9 flags §2, 6 dims §3, 3 verdicts §1, gate mins, context tags §4). taxonomy.md prose stays human source of truth; a freeze test asserts they match.
- DECISION: worst-frame pooling (§5) for temporal_stability + motion_quality is implemented as MIN over per-frame 0-4 scores (0 = worst). Mean pooling for the other 4 dims. Flags use MAX pooling across frames (any frame triggers => clip triggers).
- DECISION: CORAL ordinal loss for the 6 dimension regressors (K=5 levels, K-1=4 ordered thresholds); N/A (null) dims masked out of the loss. Verdict = 3-way softmax CE. Flags = 9 independent sigmoid + pos-weighted BCE (missing a hard-fail costs more).

## Stage 0 + samples (verified)
- Stage 0 green: `pytest tests/test_taxonomy_freeze.py tests/test_schema.py` -> 25 passed. Freeze test parses taxonomy.md §2 table (9 flag IDs+sections) and §7.1 scores keys (6 dims) and asserts equality with taxonomy_schema.py.
- Samples green: `pytest tests/test_samples.py` -> all 8 clips+sidecars build, sidecars validate as §7.1 records, duplicate is byte-identical to clean_pass. Verified durations via ffprobe; corrupted.mp4 yields 0 decodable frames (h264 "partial file").
- OPEN QUESTION: taxonomy has no dedicated flag for "below minimum duration" (§5 treats min-duration as a verdict rule, not a flag). REJECT (§7.1) requires >=1 flag OR >=1 score of 0, and duration is not a scored dimension. PLACEHOLDER: map sub-minimum-duration -> `delivery_failure` (asset cannot serve downstream). Revisit if taxonomy adds a spec/duration flag.
- DECISION: delivery_failure detection = ffprobe fails OR frame extraction yields 0 decodable frames OR decode errors. corrupted.mp4 keeps moov (faststart) so ffprobe reports duration, but mdat is truncated -> 0 frames decode; this exercises the decode-failure path.
- Perceptual-hash check: duplicate==clean (hamming 0); all other sample pairs >=26. Dedup threshold 4 => only the intended duplicate merges.
