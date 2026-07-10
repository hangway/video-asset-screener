---
name: run-stage
description: Run one stage (ingest, prelabel, annotate, dataset, train, evaluate, screen) or the whole video-asset-screener pipeline on a folder of clips. Use when the user wants to execute or re-run part of the screening pipeline, screen a new clip folder, or produce the dashboard.
---

# run-stage

Drive the `video_screener` pipeline via its `pipeline` CLI (typer).

## Stages (each independently runnable AND chainable)
`ingest → prelabel → annotate → dataset → train → evaluate → screen`

## Commands
- One stage:      `python -m video_screener.cli run <stage> --config configs/default.yaml`
- Whole pipeline: `python -m video_screener.cli run all --config configs/default.yaml`
- Screen a folder:`python -m video_screener.cli run screen --video-dir <path> --out <outdir>`
- Build dashboard:`python -m video_screener.cli dashboard --workdir runs/<name> --out runs/<name>/dashboard.html`
- (Re)make samples:`python -m video_screener.cli samples --out samples`
- Validate a file:`python -m video_screener.cli validate <file.jsonl> --kind annotation|inference`

## Notes
- All stages read/write under `workdir` (default `runs/default`), overridable with `--workdir`.
- `annotate` launches the Textual TUI interactively; add `--auto` to apply sidecar
  ground truth (samples) or accept prelabels non-interactively (used by `run all`).
- Artifacts per stage: `ingest/index.json`, `prelabel/prelabels.jsonl`,
  `annotate/annotations.jsonl`, `dataset/{splits,balance_report}.json` + `*.jsonl`,
  `train/{model.pt,train_log.json}`, `evaluate/eval_report.json`,
  `screen/{screen_results.jsonl,screen_report.html,routing.json}`.
- Config is validated by pydantic against the frozen taxonomy (invalid flag IDs /
  dimension names are rejected at load). Never edit taxonomy.md — closed vocab is law.

## Typical flow
1. Confirm samples exist (`ls samples/*.mp4`) or run `pipeline samples`.
2. `pipeline run all` to execute end to end.
3. `pipeline dashboard --workdir runs/default` and open the HTML.
4. To screen new clips: `pipeline run screen --video-dir /path/to/clips --out runs/screened`.
