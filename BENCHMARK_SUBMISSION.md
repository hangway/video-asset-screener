# Benchmark submission format

A submission is a static, reproducible report. This project does not provide a
hosted leaderboard.

## Required declaration

Include submission ID; model or checkpoint name and encoder; training dataset
versions and number of training clips; providers represented in training;
validation and test dataset versions; held-out providers; taxonomy version;
code commit SHA and random seeds; hardware; main metrics; per-verdict metrics;
per-flag precision, recall, and F1; per-dimension MAE and within-one accuracy;
provider-, aesthetic-, and motion-complexity-stratified results; calibration
status and known limitations; model license; and public weights URL when
available.

Use the machine-readable example at examples/benchmark_submission.json.

## Interpretation rules

Report missing metadata as unavailable; never fabricate a score. Explain
whether confidence is calibrated on held-out data. State whether provider
families overlap between training and evaluation. A benchmark result does not
turn the reference model into a production-certified system.

## Example metric table

The values below are placeholders and must not be read as measured results.

| Metric | Value |
| --- | ---: |
| Verdict accuracy | PLACEHOLDER |
| PASS precision / recall | PLACEHOLDER / PLACEHOLDER |
| FIX precision / recall | PLACEHOLDER / PLACEHOLDER |
| REJECT precision / recall | PLACEHOLDER / PLACEHOLDER |
| Mean dimension MAE | PLACEHOLDER |
| Calibration status | not calibrated |

Per-flag and per-dimension tables belong in the submission artifact. Review
licensing, reproducibility, and benchmark contamination before publication.
