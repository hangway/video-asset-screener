# video-asset-screener

A general-purpose, end-to-end **training + inference pipeline for screening the
usability of AI-generated video assets** (Kling, Runway, Wan, Seedance, custom
fine-tunes, …) for film/TV and short-form production workflows.

Given a folder of clips, it routes each to **PASS / FIX / REJECT** with a verdict,
6 scored quality dimensions, up to 9 canonical hard-fail flags, suggested
`fix_actions`, a confidence, and a shareable HTML report — all aligned to the
usability standard in [`taxonomy.md`](taxonomy.md).

> **Taxonomy is law.** The usability standard, closed vocabulary (9 hard-fail
> flag IDs, 6 dimensions, 3 verdicts, gate mins, aggregation rules, output
> schemas) lives in `taxonomy.md` **v0.3.1** and is the single source of truth.
> The code freezes a machine-readable projection of it in
> `video_screener/taxonomy_schema.py`; a test asserts the two never drift.
> Nothing invents or modifies a flag ID or schema field.

---

## Install

Requires Python ≥ 3.10 and `ffmpeg`/`ffprobe` on `PATH`.

```bash
# ffmpeg (Debian/Ubuntu)
sudo apt-get install -y ffmpeg

# the package (editable)
pip install -e .
# dev extras (pytest): pip install -e ".[dev]"
```

The default per-frame encoder is **offline and deterministic** (no weights to
download). To use a real CLIP/SigLIP image tower instead, install the `clip`
extra (`pip install -e ".[clip]"`) and set `model.encoder: "clip:ViT-B-32"` —
it falls back to the deterministic encoder if weights aren't available.

## Quickstart — one clip folder → screened output

```bash
# 1. (optional) synthesize the demo clips
pipeline samples --out samples

# 2. run the whole pipeline end to end on samples/
pipeline run all --config configs/default.yaml

# 3. build the dashboard and open it
pipeline dashboard --workdir runs/default --out runs/default/dashboard.html

# 4. screen a NEW folder of clips -> PASS/FIX/REJECT + HTML report
pipeline run screen --video-dir /path/to/your/clips --out runs/screened
open runs/screened/screen_report.html          # per-clip routing + reasons
```

`pipeline` is installed as a console script; you can also call it as
`python -m video_screener.cli`.

## Local open-model generation

The optional generation layer connects to a local or LAN ComfyUI server. It
does not require a hosted-provider SDK or API key. Export a workflow with
ComfyUI's **Save (API Format)** command, then provide a small bindings file
that identifies the workflow inputs the application may change:

```json
{
  "prompt": {"node_id": "6", "input_name": "text"},
  "negative_prompt": {"node_id": "7", "input_name": "text"},
  "seed": {"node_id": "3", "input_name": "seed"},
  "width": {"node_id": "5", "input_name": "width"},
  "height": {"node_id": "5", "input_name": "height"},
  "references": {}
}
```

The node IDs are workflow-specific. Generate two reproducible image
candidates with:

```bash
pipeline generate image \
  --workflow workflows/image-api.json \
  --bindings workflows/image-bindings.json \
  --prompt "A consistent character portrait" \
  --model-id local/image-model \
  --model-revision sha256:MODEL_HASH \
  --model-license Apache-2.0 \
  --seed 42 --candidates 2 --out runs/generation
```

`pipeline generate video` uses the same contract and additionally supports
FPS, duration/frame-count bindings, first/last keyframes, and role-based
references (`--reference character=/path/to/portrait.png`). Each run writes
`generation.json` beside its candidates with the prompt, references, model
revision, license, workflow version, seed, hardware metadata, hashes, runtime,
and output paths. ComfyUI should remain on a trusted local network; it is not
an internet-facing authentication boundary.

## Multi-engine workflow architecture

The application now has a portable, node-based workflow contract inspired by
FlowPix's separation of canvas authoring, personal workflows, reusable
templates, community metadata, and generation history. The contract is
backend-neutral: a future visual canvas and the CLI can edit the same JSON or
YAML graph without coupling the application to ViMax, ComfyUI, the screener,
or OpenMontage internals.

The built-in template connects the complete production path:

```text
project inputs -> ViMax plan -> local image generation -> image gate
  -> local video generation -> clip gate -> OpenMontage assembly
  -> final delivery gate -> video + quality report + edit timeline
```

Validate or inspect it before connecting engine-specific adapters:

```bash
pipeline workflow validate configs/workflows/vimax-screen-openmontage.json
pipeline workflow inspect configs/workflows/vimax-screen-openmontage.json
```

Each manifest records typed node ports, edges, canvas positions, public input
and output slots, engine and model requirements, visibility, use cases,
categories, and tags. Validation rejects missing endpoints, incompatible media
types, ambiguous single-input wiring, unconnected required inputs, and cycles.
The existing `generation.json` manifests provide the durable source for a
future generation-history view; no hosted model provider is required.

## The 7 stages

Each stage is independently runnable *and* chainable; artifacts land under
`--workdir` (default `runs/default`).

| # | Stage | What it does | Key artifact |
|---|-------|--------------|--------------|
| 1 | **ingest** | scan dirs, ffprobe, sample frames per §5 (1 fps + scene changes; <4 s at 0.5 s), perceptual-hash clip dedup, decode-failure detection | `ingest/index.json` |
| 2 | **prelabel** | auto pre-annotation: sharpness/exposure from frame metrics, brightness-flicker temporal heuristic, objective `delivery_failure`; emits §7.1 records (`needs_human_review=True`) | `prelabel/prelabels.jsonl` |
| 3 | **annotate** | Textual TUI for human review — frame thumbnails, verdict/flag/score editing, closed-vocab + REJECT-needs-reason enforcement; `--auto` applies sidecar ground truth | `annotate/annotations.jsonl` |
| 4 | **dataset** | leakage-free train/val/test splits (group by phash near-dup cluster + source id), class-balance report, training export | `dataset/splits.json`, `dataset/balance_report.json` |
| 5 | **train** | multi-task model: frozen per-frame features → temporal transformer → 3 heads (verdict softmax, 6 CORAL ordinal dims, 9 pos-weighted sigmoid flags); config-driven, resumable | `train/model.pt`, `train/train_log.json` |
| 6 | **evaluate** | verdict confusion matrix, per-dimension MAE/exact/±1, per-flag P/R/F1, stratified by aesthetic_family + motion_complexity, worst-failure gallery, **verdict-vs-flags/dims consistency rate** | `evaluate/eval_report.json` |
| 7 | **screen** | batch inference: objective delivery-failure override → predicted-flag override → head-primary verdict with §1 PASS-gate; §7.2 output + shareable HTML report | `screen/screen_results.jsonl`, `screen/screen_report.html` |

Run one stage: `pipeline run <stage> --config configs/default.yaml`.

## Model architecture (stage 5)

```
frozen per-frame features (deterministic encoder, or CLIP/SigLIP if local)
        │            [T, feature_dim]  — frozen, cached to disk
        ▼
lightweight temporal transformer (2–4 layers)      → per-frame contextual embeddings
        │
        ├── verdict head  → 3-way softmax (cross-entropy)          [grounded in dims+flags]
        ├── 6 dimension heads → CORAL ordinal (rank-consistent); N/A dims masked from loss
        └── 9 flag heads  → independent sigmoid + pos-weighted BCE (a missed hard-fail
                            costs more than a false alarm)
```

Clip-level pooling matches **taxonomy §5**: `temporal_stability` &
`motion_quality` use the **worst frame** (min); other dimensions use the **mean**;
hard-fail flags use the **max** (any frame triggering triggers the clip).

## Taxonomy version pinning

- `video_screener/taxonomy_schema.TAXONOMY_VERSION` pins the frozen version
  (**0.3.1**). Every artifact records its `taxonomy_version`.
- Configs must pin the same version (`taxonomy_version: "0.3.1"`) or loading fails.
- `tests/test_taxonomy_freeze.py` parses `taxonomy.md` §2/§3/§7 and asserts the
  frozen constants (9 flag IDs, 6 dimensions, gate mins, verdicts) still match
  the prose. Bump the prose → bump the code → re-run the freeze test.

## Configuration

YAML validated by pydantic (`video_screener/config.py`); invalid flag IDs or
dimension names are rejected at load. See `configs/default.yaml`. Override the
workdir or inputs on the CLI: `--workdir`, `--video-dir`.

## Output schemas (the data contract)

- **§7.1 annotation** (`AnnotationRecord`): full training/eval label — verdict,
  scores, flags, context tags, fix_actions, reject_reason, frame_evidence. Rules
  enforced: REJECT ⇒ reason + (≥1 flag OR a dim=0); FIX ⇒ fix_actions; PASS ⇒ no flag.
- **§7.2 inference** (`InferenceRecord`): screen output — verdict, confidence,
  flags, sparse scores, fix_actions, primary_reasons, needs_human_review.

Validate any file: `pipeline validate <file.jsonl> --kind annotation|inference`.

## Metrics backends (the measurement layer)

Technical dimension scoring (sharpness / exposure / temporal stability) reads
per-frame measurements through one pluggable interface, selected by
`metrics_backend` in the config:

- **`ffprobe`** (default): ffmpeg's `signalstats` + `blurdetect` filters in a
  single ffprobe call per clip — the standardized per-frame signals of
  **QCTools lineage** (the archive/broadcast QC toolchain built on these same
  filters, credit to the QCTools / BAVC community):
  - `YAVG` — average luma (0–255): exposure level
  - `YDIF` — mean absolute luma change vs the previous frame: flicker /
    temporal instability
  - `YLOW` / `YHIGH` — 10th / 90th luma percentiles: crushed shadows /
    blown highlights (our clipping proxy counts *mostly-crushed* frames)
  - `blur` — blurdetect edge-width blurriness (higher = blurrier)
  - `TOUT` — temporal-outlier pixel fraction (dropout/noise candidates;
    parsed and reported, reserved for future compression-artifact work
    alongside ffmpeg's `blockdetect` blockiness filter)
- **`opencv`**: the original hand-rolled statistics over sampled frames
  (variance-of-Laplacian sharpness, mean-luma exposure, brightness-std
  flicker). Kept as the fallback; the ffprobe backend also degrades to it
  per clip when a probe fails or ffprobe is missing.

The scales differ between backends (YDIF ≠ brightness-std; blurdetect is
inverted vs Laplacian variance), so thresholds are **backend-scoped** in the
config (`prelabel.ffprobe.*`) and were calibrated by measuring the sample set
(see `notes.md`). A parity test asserts both backends produce identical
prelabel verdicts on `samples/`; that gate is what justified the default flip.

Ingest additionally runs `freezedetect` + `blackdetect` per clip (one ffprobe
call) and records frozen-video / black intervals as objective evidence:
full-coverage intervals (≥90% of duration) raise the existing
`delivery_failure` flag (§2.8 — no usable content); partial intervals become
`temporal_stability` frame evidence for the human annotate stage. This
upgrades the measurement backend only — taxonomy v0.3.1 scoring semantics,
gates, and flags are unchanged, and the embedding encoder is untouched.

## Reference consistency & edge stability (screen stage)

Give the screen stage reference images of your subjects and it checks every
clip against them — wired entirely into the **existing**
`reference_inconsistency` hard-fail flag (taxonomy v0.3.1 unchanged):

```bash
pipeline run screen --video-dir clips/ --reference-dir refs/ --out runs/screened
# refs/
#   subject_a/  img1.png img2.jpg ...   # one subdirectory per subject
#   subject_b/  ...
#   loose.png                           # files at the top level -> subject "default"
```

- **Clip-vs-reference score**: every (sampled) frame's embedding is compared to
  the assigned subject's references — best-match cosine per frame, aggregated
  **worst-frame** per §5 (min over frames; the subject is whichever scores
  highest). A clip below `consistency.min_reference_similarity` raises
  `reference_inconsistency` and REJECTs through the normal flag path.
- **Within-clip drift** (auxiliary): max consecutive-frame cosine distance.
  Above `consistency.max_frame_drift` the clip is a morphing candidate and gets
  `needs_human_review` — drift alone never changes a verdict.
- **Head/tail edge stability**: per-frame similarity + drift are analyzed
  separately for the head window, tail window (`consistency.edge_window_sec`)
  and clip body. An edge that is a statistical outlier vs the body (beyond
  `consistency.edge_outlier_sigma` body standard deviations) emits a
  `trim_head`/`trim_tail` fix_action with a suggested trim duration and routes
  **FIX** per §1 (trims are minor allowed fixes) — never REJECT.
- **Outputs**: `consistency_report.json` (per-clip scores, per-frame worst
  offenders, drift, edge analysis) and, in `screen_report.html`, a reference
  gallery plus a per-subject ranking table (best → worst) with
  inconsistency / drift / trim markers.

References are embedded by the **same pluggable encoder** as clip frames and
cached (`reference_cache/`, keyed by encoder + content digest). NOTE: the
default deterministic encoder captures technical statistics, not identity —
it verifies the mechanism but cannot judge whether a face/character matches;
use a real CLIP/SigLIP encoder for meaningful consistency scores.

## Skills

Common operations are wrapped as Claude Code skills in `.claude/skills/`:
`run-stage`, `inspect-dataset`, `compare-runs`.

## Testing

```bash
pytest -q          # ingest sampling+dedup, schema validation, split leakage,
                   # eval metric math, model/CORAL/pooling, screen contract, TUI, dashboard
```

## Boundaries

Video only. No still-image pipeline, no auth, no cloud, no web server. Deliverable
is this pip-installable package + README.

**Audio is explicitly OUT OF SCOPE for v0.3.x**: clips are judged on visual
usability alone — no audio decoding, sync, or loudness checks anywhere in the
pipeline (a future taxonomy revision would have to introduce them; nothing in
v0.3.1 scores sound).

## Notes / caveats

- The bundled `samples/` set is **8 tiny synthetic clips** for exercising the
  pipeline end to end — a functional demo, not an accuracy benchmark. With so few
  clips, held-out metrics are thin and some labels (e.g. the `watermark`
  hard-fail) have no positive training example. See `notes.md` for the running
  log of decisions, verified results, and open questions.
- Screen confidence is the probability the *deciding source* assigns to the
  emitted verdict (objective delivery rule → 1.0; flag-forced REJECT →
  strongest triggered flag's sigmoid; verdict-head routing → head softmax of
  the emitted verdict). It is **not** calibrated on a held-out set with this
  toy dataset (documented placeholder per §7.2 — calibrate on a real
  validation set before production use).
- Reference-consistency scores from the **deterministic encoder are
  structural only** — it cannot judge subject identity. Swap in a CLIP/SigLIP
  encoder (`model.encoder: "clip:ViT-B-32"`, weights available locally) before
  trusting consistency verdicts on real footage.
