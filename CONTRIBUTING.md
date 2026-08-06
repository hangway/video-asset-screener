# Contributing

Thank you for helping make video-asset-screener a useful, inspectable community
framework. The project accepts small, well-documented contributions from people
working with different generators, providers, styles, and editorial workflows.

## Contribution areas

1. Code and bug fixes: keep changes small, tested, and compatible.
2. Documentation: improve setup instructions, examples, and explanations.
3. Taxonomy anchors: propose diverse, licensed examples for existing dimensions
   and flags. Do not add a canonical flag in an ordinary pull request.
4. Community video datasets: contribute metadata first, then media only when
   redistribution is explicitly allowed. See DATA_CONTRIBUTION.md.
5. Annotation corrections: provide evidence and explain the requested verdict,
   scores, flags, and fix actions.
6. Encoders and feature extractors: document dependencies, weights, provenance,
   feature dimensions, and limitations.
7. Training checkpoints: include code revision, dataset versions, seeds,
   hardware, and model license.
8. Evaluation and benchmark submissions: follow BENCHMARK_SUBMISSION.md.
9. New QC metrics: explain measurement, calibration data, failure modes, and
   relationship to taxonomy v0.3.1.
10. Platform-specific installation documentation: document OS, Python, ffmpeg,
    accelerator, and dependency versions actually tested.

## Development setup

Requires Python 3.10 or newer, ffmpeg, and ffprobe.

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest -q

On Windows, use .venv\\Scripts\\Activate.ps1. The optional CLIP extra is
installed with python -m pip install -e ".[clip]"; it is not required for the
deterministic demo pipeline.

## Branches, commits, and pull requests

- Branch from main and use a focused branch name.
- Keep commits coherent and explain behavior changes in the commit message.
- Add or update tests for code and schema changes.
- Run pytest -q and the end-to-end demo pipeline before opening a PR.
- Keep generated artifacts, local runs, credentials, and large media out.
- Describe backward-compatibility impact and migration steps.
- Include screenshots or sample output when changing the TUI, dashboard, or CLI.
- Request review from someone able to assess both code and data risks.

## Dataset, model, and media licensing

Every dataset or model contribution must document media and annotation licenses,
redistribution permission, source URL or safe relative path, checksum, and
provenance. Model contributions must also document a model license and public
weights URL when applicable.

Do not contribute private, copyrighted, unsafe, or non-redistributable data. Do
not commit celebrity faces, branded IP, personal data, leaked credentials,
private prompts, or reference images that cannot legally be shared. Metadata can
describe unavailable prompts or references explicitly; never guess them.

## Taxonomy and schema rules

Taxonomy v0.3.1 remains the source of truth for this release. Changing a
verdict, scored dimension, hard-fail flag, gate, or canonical meaning requires
a formal taxonomy version bump and migration plan. No contributor may invent a
new hard-fail ID in an ordinary pull request. Existing annotation and inference
records must continue to load; new fields should be optional.

## Review standard

A good contribution is reproducible, scoped, evidence-backed, and candid about
limitations. Reviewers may request a smaller change, additional fixtures, or a
license and safety review. A passing test suite is necessary but does not prove
a model is accurate or a dataset is benchmark-ready.
