# /goal — AI Video Asset Usability Screening Pipeline (synced to taxonomy v0.3)

## Goal
Transform the existing fine-tuning work in this repo into a sellable, end-to-end
**training + inference pipeline for screening usable AI-generated video assets**
(Kling, Runway, Wan, Seedance, custom fine-tunes, etc.) in film/TV and
short-form production workflows.

The usability standard is defined in `taxonomy.md` (v0.3) at repo root — the
source of truth for the verdict (PASS/FIX/REJECT), the 9 canonical hard-fail
flag IDs, all scored dimensions, gate mins, frame-sampling and aggregation
rules, and the annotation output schema. Do NOT invent or modify any label,
flag ID, or schema field; if a stage needs something taxonomy.md doesn't
define, record it in `notes.md` as an open question and use a placeholder.

## The 7 stages (each independently runnable AND chainable)
1. **ingest** — scan video dirs, extract frames per taxonomy §5 sampling rules
   (1 fps + scene changes via ffmpeg/scenedetect; <4s clips at 0.5s),
   perceptual-hash dedup at clip level, build asset index
2. **prelabel** — auto pre-annotation: technical dims (sharpness, exposure)
   from frame metrics; temporal-stability heuristics (frame-diff/flicker);
   optional MLLM-assisted prelabels for prompt fidelity — all emitting the
   taxonomy annotation schema with canonical flag IDs only
3. **annotate** — TUI for human review/correction: shows sampled frames as
   thumbnails (terminal-image or browser handoff), plays clip via system
   player, enforces closed flag vocabulary and REJECT-must-have-reason rule
4. **dataset** — build train/val/test splits (no near-dup or same-source
   leakage across splits), class balance report per verdict and per flag,
   export in the training format the repo already uses
5. **train** — wrap existing fine-tuning code as a multi-task model
   (verdict + dimension scores + multi-label flags); config-driven; resumable
6. **evaluate** — per-dimension metrics, verdict confusion matrix, per-flag
   precision/recall, stratified by aesthetic_family and motion_complexity
   context tags, worst-failure gallery
7. **screen** — batch inference CLI: point at a folder of clips, get
   PASS/FIX/REJECT routing with reasons, suggested fix_actions, confidence,
   plus a shareable HTML report

## Done means ALL of:
- [ ] `pipeline run <stage>` and `pipeline run all` work end to end on
      `samples/` (create tiny synthetic clips via ffmpeg if absent —
      include at least one per hard-fail category that's synthesizable:
      watermark overlay, corrupted file, flicker injection, duplicate)
- [ ] TUI (stage 3) launches, navigates, displays frames, saves annotations
      that validate against the taxonomy schema
- [ ] HTML dashboard: reads run artifacts (JSON), shows stage status,
      eval metrics, and a dataset viewer with clip thumbnails +
      per-clip verdict/flags — single file, no server
- [ ] Common operations wrapped as skills in `.claude/skills/`
      (minimum: run-stage, inspect-dataset, compare-runs)
- [ ] pytest covers: ingest sampling + dedup, schema validation of
      annotations, split integrity (leakage checks), eval metric math,
      screen-CLI output contract. All tests pass.
- [ ] README: install, quickstart (one clip folder → screened output),
      stage docs, taxonomy version pinning

## Boundaries
- Video only. No still-image pipeline in this scope.
- Don't refactor code unrelated to the 7 stages.
- Don't add features beyond this brief (no auth, no cloud, no web server).
- Don't touch, reinterpret, or extend taxonomy.md — closed vocabulary is law.
- Deliverable form: pip-installable package + README. Nothing else.

## Verification discipline
- Run tests yourself after every stage; fix before moving on. Don't ask me.
- Before reporting ANY stage complete, audit the claim against an actual
  command/test run from this session. Unverified = say so explicitly.
- Keep `notes.md`: one lesson per line, decisions logged, wrong entries deleted.
- Final report: per-stage status, evidence pointer (test/command output),
  open questions list.
