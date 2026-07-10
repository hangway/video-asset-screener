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

## Stage 1 ingest (verified)
- `pytest tests/test_ingest.py` -> 8 passed. `pipeline run ingest` on samples -> 8 assets, 7 unique, 1 duplicate (duplicate->clean_pass), 1 decode failure (corrupted, 0 frames).
- §5 sampling verified: 5s@1fps -> 5 frames; 2s -> every 0.5s -> 4 frames; 0.29s -> 1 frame.
- Suppressed OpenCV/libav decode-error spam via OPENCV_FFMPEG_LOGLEVEL=-8 + cv2.setLogLevel(0).
- Scene detection is best-effort (try/except -> [] on failure); synthetic clips have no hard cuts so uniform sampling dominates.

## Stage 2 prelabel (verified)
- `pytest tests/test_prelabel.py` -> 14 passed. Prelabel verdicts match sidecar ground truth on 7/8 samples; watermark (PASS vs REJECT) is the documented exception — subjective flags are deferred to human annotate stage.
- Calibrated metrics on real sample frames: sharpness var-of-laplacian bins [60,150,400,1200]; exposure from brightness+clip-fraction; FLICKER = std of per-frame mean brightness (isolates brightness pumping from motion; clean mandelbrot motion gave misleading spatial frame-diff).
- Fixed lowres sample: nearest-neighbor upscale gave HIGH laplacian (blocky); switched to bilinear+boxblur -> genuinely soft (var ~11 -> sharpness 0). Fixed flicker: 8Hz aliased at 1fps sampling -> lowered to 0.35Hz (brightness_std 37 vs clean 9.5).
- DECISION: prelabel scope = sharpness/exposure/temporal + delivery_failure(objective). composition/motion = neutral prior 3; prompt_fidelity = null (no prompt offline); every record needs_human_review=True. Verdict is a heuristic starting point; annotate stage sets ground truth.
- aggregate.py = shared §1/§5/§6 verdict derivation + §5 pooling (worst=min for temporal/motion, mean for rest, max for flags) + consistency_violations() for evaluate.
