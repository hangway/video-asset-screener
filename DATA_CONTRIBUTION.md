# Contributing AI-generated video data

This guide covers metadata and media contributions to the community dataset. The
repository accepts metadata without accepting the underlying video. No large
video files are required in a pull request.

## Contribution lifecycle

submitted -> schema validated -> license checked -> human reviewed ->
accepted -> benchmark eligible

A submission can remain community-submitted indefinitely. Not every submitted
clip becomes benchmark data.

## Contribution tiers

- community-submitted: received and schema-valid, but not yet reviewed.
- reviewed: a human checked metadata, provenance, license, and annotation
  evidence.
- benchmark-eligible: reviewed data with explicit redistribution rights,
  declared licenses, and a reviewed annotation path. Eligibility is a
  governance decision, not a model-quality claim.

Use pipeline validate-manifest path/to/manifest.jsonl before opening a PR.
Validation is metadata-only and never downloads remote videos.

## Required record fields

Each row must declare clip ID, provider, model, generation mode, duration,
width, height, frame rate when available, SHA-256, and either a public URL or a
safe local relative path (never both). It must also declare whether a prompt
and reference images are available, with paths when shareable; asset role, shot
type, content category, aesthetic family, motion complexity; annotation path
when reviewed; media and annotation licenses; explicit redistribution
permission; and risk declarations for real people, logos or trademarks,
protected characters, and personal data.

The machine-readable schema is in video_screener/community_manifest.py.
Context tags are closed against the existing taxonomy vocabulary so data cannot
silently introduce a new category.

## Prompts and references that cannot be shared

Set prompt_available or reference_images_available to false when an artifact
cannot be redistributed. Omit the corresponding path. Do not copy proprietary
prompts, private URLs, or reference images into the repository. A missing
artifact is valid metadata and must remain explicit.

## Media and annotation safety

Confirm that media can be redistributed under the declared license. Do not
submit private people, personal data, unsafe material, copyrighted characters,
logos, or watermarked material unless legally cleared and accurately declared.
Reviewers may request removal or a metadata-only contribution.

## Review expectations

Reviewers check schema validity, provenance, licensing, privacy, duplicates, and
whether annotations cite concrete frame evidence. Benchmark releases use a
versioned holdout policy and are not silently rewritten after publication.
