# Dataset governance

This document defines how community metadata and media become release data.

## Admission process

1. A contributor submits a manifest row and provenance.
2. Automation validates JSONL structure, taxonomy context, checksums, and status.
3. A reviewer checks licensing, redistribution permission, safety, privacy,
   duplicates, and annotation evidence.
4. Maintainers assign reviewed or benchmark-eligible status only when required.
5. A release records an immutable manifest, split definition, checksums,
   taxonomy version, and review notes.

## Review roles

- Contributor: owns provenance and declares licenses and risks.
- Metadata reviewer: checks schema, provider/model fields, and completeness.
- Safety and privacy reviewer: checks personal data, unsafe content, logos,
  trademarks, and protected characters.
- Annotation reviewer: checks evidence, verdict rules, scores, and flags.
- Release maintainer: approves versioned releases and holdout policy.

One person may fill multiple roles for a small submission, but benchmark
releases should use independent review where practical.

## Licensing, safety, and duplicates

Media and annotations need explicit licenses. Benchmark eligibility requires
redistribution permission for both media and annotations. Duplicate media should
be detected by checksum and perceptual similarity; duplicates must not inflate
class or provider balance. Unsafe or legally unclear records are rejected or
kept as metadata-only records.

## Balance and benchmark holdouts

Release reports should show PASS/FIX/REJECT balance, hard-fail frequencies,
provider and model distribution, aesthetic family, motion complexity, asset
role, missing prompt/reference rates, licenses, and contribution status.
Benchmark test sets should hold out providers or model families when possible
and disclose train/test overlap. A benchmark release must not silently modify
its holdout records after publication.

## Versioning and corrections

Dataset releases are immutable snapshots with a version, manifest checksum,
taxonomy version, and release notes. Corrections create a new release or a
documented patch release. Removal requests, annotation disputes, and privacy
concerns are tracked with the affected clip ID and resolved with an audit note.
Problematic records may be deprecated without rewriting history.

## Attribution and contamination

Preserve contributor attribution and source provenance. Do not train or
evaluate on benchmark test records unless the submission explicitly declares
contamination. A model result without disclosed training/test overlap is not a
valid benchmark comparison.
