"""Schema and validation helpers for community dataset manifests.

The manifest is deliberately metadata-only: validation never downloads media.
It records provenance, licensing, risk declarations, and the review lifecycle
needed to turn community submissions into benchmark-eligible data.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator, model_validator

from .taxonomy_schema import CONTEXT_TAG_VALUES, TAXONOMY_VERSION

ContributionStatus = Literal["community-submitted", "reviewed", "benchmark-eligible"]
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CommunityMedia(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl | None = None
    relative_path: str | None = None
    sha256: str
    duration_sec: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float | None = Field(default=None, gt=0)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be exactly 64 hexadecimal characters")
        return value.lower()

    @field_validator("relative_path")
    @classmethod
    def _relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must be relative and cannot contain '..'")
        if not value.strip():
            raise ValueError("relative_path cannot be empty")
        return value

    @model_validator(mode="after")
    def _one_media_locator(self) -> "CommunityMedia":
        has_url = self.url is not None
        has_path = bool(self.relative_path)
        if has_url == has_path:
            raise ValueError("exactly one of media.url or media.relative_path must be usable")
        if self.url is not None and urlparse(str(self.url)).scheme not in {"http", "https"}:
            raise ValueError("media.url must use http or https")
        return self


class CommunityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_available: bool
    reference_images_available: bool
    prompt_path: str | None = None
    reference_manifest_path: str | None = None

    @model_validator(mode="after")
    def _explicit_absence(self) -> "CommunityProvenance":
        if not self.prompt_available and self.prompt_path is not None:
            raise ValueError("prompt_path must be omitted when prompt_available is false")
        if not self.reference_images_available and self.reference_manifest_path is not None:
            raise ValueError(
                "reference_manifest_path must be omitted when "
                "reference_images_available is false"
            )
        return self


class CommunityContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_role: str
    shot_type: str
    content_category: str
    aesthetic_family: str
    motion_complexity: str

    @model_validator(mode="after")
    def _taxonomy_values(self) -> "CommunityContext":
        for field_name in (
            "asset_role",
            "shot_type",
            "content_category",
            "aesthetic_family",
            "motion_complexity",
        ):
            value = getattr(self, field_name)
            allowed = CONTEXT_TAG_VALUES[field_name]
            if value not in allowed:
                raise ValueError(
                    f"context.{field_name}={value!r} is not a valid taxonomy context tag; "
                    f"expected one of {list(allowed)}"
                )
        return self


class CommunityLicenses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_license: str | None = None
    annotation_license: str | None = None
    redistribution_allowed: bool

    @field_validator("media_license", "annotation_license")
    @classmethod
    def _nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("license declarations cannot be blank")
        return value


class CommunityRiskDeclarations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_real_person: bool
    contains_logo_or_trademark: bool
    contains_protected_character: bool
    contains_personal_data: bool


class CommunityManifestRecord(BaseModel):
    """One metadata row in a community dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    taxonomy_version: str = TAXONOMY_VERSION
    clip_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    generation_mode: str = Field(min_length=1)
    media: CommunityMedia
    provenance: CommunityProvenance
    context: CommunityContext
    annotation_path: str | None = None
    licenses: CommunityLicenses
    risk_declarations: CommunityRiskDeclarations
    contribution_status: ContributionStatus

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, value: str) -> str:
        if value != "0.1.0":
            raise ValueError("community manifest schema_version must be '0.1.0'")
        return value

    @field_validator("taxonomy_version")
    @classmethod
    def _taxonomy_version(cls, value: str) -> str:
        if value != TAXONOMY_VERSION:
            raise ValueError(
                f"community manifest taxonomy_version must remain {TAXONOMY_VERSION!r}"
            )
        return value

    @field_validator("annotation_path")
    @classmethod
    def _annotation_path(cls, value: str | None) -> str | None:
        if value is not None:
            if not value.strip():
                raise ValueError("annotation_path cannot be blank")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("annotation_path must be a safe relative path")
        return value

    @model_validator(mode="after")
    def _eligibility(self) -> "CommunityManifestRecord":
        if self.contribution_status == "benchmark-eligible":
            if not self.annotation_path:
                raise ValueError(
                    "benchmark-eligible records require reviewed annotations "
                    "via annotation_path"
                )
            if not self.licenses.media_license:
                raise ValueError(
                    "benchmark-eligible records require a declared media_license"
                )
            if not self.licenses.annotation_license:
                raise ValueError(
                    "benchmark-eligible records require a declared annotation_license"
                )
            if self.licenses.redistribution_allowed is not True:
                raise ValueError(
                    "benchmark-eligible records require explicit redistribution_allowed=true"
                )
        return self


def validate_manifest_rows(path: str | Path) -> tuple[list[CommunityManifestRecord], list[dict[str, Any]]]:
    """Validate every JSONL row and return valid records plus line errors."""

    records: list[CommunityManifestRecord] = []
    errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            records.append(CommunityManifestRecord.model_validate(payload))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    return records, errors


def manifest_summary(records: list[CommunityManifestRecord]) -> dict[str, Any]:
    """Return deterministic distributions for CLI output and reports."""

    def counts(values: list[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    return {
        "records": len(records),
        "contribution_status": counts([r.contribution_status for r in records]),
        "providers": counts([r.provider for r in records]),
        "models": counts([r.model for r in records]),
        "media_licenses": counts([
            r.licenses.media_license or "(undeclared)" for r in records
        ]),
        "annotation_licenses": counts([
            r.licenses.annotation_license or "(undeclared)" for r in records
        ]),
        "missing_prompt": sum(not r.provenance.prompt_available for r in records),
        "missing_references": sum(
            not r.provenance.reference_images_available for r in records
        ),
        "taxonomy_version": TAXONOMY_VERSION,
    }
