# Roadmap

No dates are promised. Each phase should preserve taxonomy v0.3.1 until a formal
taxonomy release is approved.

## Phase 1: framework and schema stabilization

Harden the community manifest, backward-compatible artifacts, validation
commands, contributor documentation, and deterministic examples.

## Phase 2: community dataset collection

Collect diverse metadata and licensed clips across providers, styles, roles, and
motion complexity, with explicit review status and provenance.

## Phase 3: inter-annotator agreement analysis

Measure disagreement, refine annotation guidance, and document ambiguous
FIX/REJECT cases without changing canonical meanings silently.

## Phase 4: provider-holdout benchmarks

Publish versioned train/validation/test splits with held-out providers and
contamination checks.

## Phase 5: calibrated community reference models

Train reproducible experimental checkpoints and calibrate confidence on
sufficient held-out data. Accuracy claims remain tied to released data.

## Phase 6: optional multimodal fusion

Explore prompt, reference, and video fusion only when licensing and privacy
permit, with explicit ablations and failure analysis.
