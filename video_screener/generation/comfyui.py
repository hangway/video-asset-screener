"""Local ComfyUI workflow adapter for image and video generation.

ComfyUI workflows are model-specific. This adapter only patches explicitly
declared node inputs, submits API-format workflows, waits for history, and
persists the resulting candidates plus a neutral generation manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import (
    MAX_SEED,
    GenerationArtifact,
    GenerationResult,
    GenerationStatus,
    ImageGenerationRequest,
    MediaKind,
    VideoGenerationRequest,
    write_generation_result,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
OUTPUT_KEYS = ("images", "gifs", "videos", "files")


class ComfyUIError(RuntimeError):
    """Expected connection, workflow, or generation failure."""


class WorkflowBindingError(ComfyUIError):
    """A declared request field cannot be bound to the workflow."""


class ComfyUIConnection(BaseModel):
    """Connection and polling policy for a local or LAN ComfyUI server."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8188"
    api_prefix: str = ""
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    job_timeout_seconds: float = Field(default=1800.0, gt=0.0)
    poll_interval_seconds: float = Field(default=0.5, gt=0.0)

    @field_validator("base_url")
    @classmethod
    def _valid_http_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        return value

    @field_validator("api_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        value = value.strip("/")
        return f"/{value}" if value else ""


class WorkflowInputBinding(BaseModel):
    """One request value mapped to one ComfyUI node input."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    input_name: str = Field(min_length=1)


class ComfyUIWorkflowBindings(BaseModel):
    """Model-independent input map for an API-format ComfyUI workflow."""

    model_config = ConfigDict(extra="forbid")

    prompt: WorkflowInputBinding
    seed: WorkflowInputBinding
    width: WorkflowInputBinding
    height: WorkflowInputBinding
    negative_prompt: WorkflowInputBinding | None = None
    duration_seconds: WorkflowInputBinding | None = None
    fps: WorkflowInputBinding | None = None
    frame_count: WorkflowInputBinding | None = None
    first_frame: WorkflowInputBinding | None = None
    last_frame: WorkflowInputBinding | None = None
    references: dict[str, WorkflowInputBinding] = Field(default_factory=dict)


def load_comfyui_workflow(path: str | Path) -> dict[str, Any]:
    """Load and validate a workflow exported with ComfyUI's API format."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ComfyUI workflow {source}: {exc}") from exc
    return validate_comfyui_workflow(raw)


def load_comfyui_bindings(path: str | Path) -> ComfyUIWorkflowBindings:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ComfyUI bindings {source}: {exc}") from exc
    return ComfyUIWorkflowBindings.model_validate(raw)


def validate_comfyui_workflow(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("ComfyUI workflow must be a non-empty JSON object")
    workflow = {str(node_id): copy.deepcopy(node) for node_id, node in raw.items()}
    malformed = [
        node_id
        for node_id, node in workflow.items()
        if not isinstance(node, dict)
        or not isinstance(node.get("inputs"), dict)
        or not isinstance(node.get("class_type"), str)
    ]
    if malformed:
        raise ValueError(
            "ComfyUI workflow must use API format; malformed nodes: "
            + ", ".join(malformed[:8])
        )
    return workflow


def _bind(
    workflow: dict[str, Any], binding: WorkflowInputBinding, value: Any
) -> None:
    node = workflow.get(binding.node_id)
    if node is None:
        raise WorkflowBindingError(
            f"workflow node {binding.node_id!r} does not exist"
        )
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or binding.input_name not in inputs:
        raise WorkflowBindingError(
            f"workflow node {binding.node_id!r} has no input "
            f"{binding.input_name!r}"
        )
    inputs[binding.input_name] = value


def render_comfyui_workflow(
    template: dict[str, Any],
    bindings: ComfyUIWorkflowBindings,
    request: ImageGenerationRequest | VideoGenerationRequest,
    *,
    seed: int,
    uploaded_references: dict[str, str] | None = None,
    uploaded_first_frame: str | None = None,
    uploaded_last_frame: str | None = None,
) -> dict[str, Any]:
    """Return one fully-bound workflow copy for a generation candidate."""
    workflow = validate_comfyui_workflow(template)
    _bind(workflow, bindings.prompt, request.prompt)
    _bind(workflow, bindings.seed, seed)
    _bind(workflow, bindings.width, request.width)
    _bind(workflow, bindings.height, request.height)

    if bindings.negative_prompt is not None:
        _bind(workflow, bindings.negative_prompt, request.negative_prompt)
    elif request.negative_prompt:
        raise WorkflowBindingError(
            "request has a negative prompt but the workflow has no binding"
        )

    uploaded_references = uploaded_references or {}
    for reference in request.references:
        binding = bindings.references.get(reference.role)
        if binding is None:
            raise WorkflowBindingError(
                f"workflow has no binding for reference role {reference.role!r}"
            )
        server_path = uploaded_references.get(reference.role)
        if not server_path:
            raise WorkflowBindingError(
                f"reference role {reference.role!r} was not uploaded"
            )
        _bind(workflow, binding, server_path)

    if isinstance(request, VideoGenerationRequest):
        if bindings.fps is None:
            raise WorkflowBindingError("video workflow requires an fps binding")
        _bind(workflow, bindings.fps, request.fps)
        if bindings.frame_count is not None:
            _bind(workflow, bindings.frame_count, request.frame_count)
        elif bindings.duration_seconds is not None:
            _bind(workflow, bindings.duration_seconds, request.duration_seconds)
        else:
            raise WorkflowBindingError(
                "video workflow requires frame_count or duration_seconds binding"
            )

        if request.first_frame is not None:
            if bindings.first_frame is None or not uploaded_first_frame:
                raise WorkflowBindingError(
                    "first_frame was supplied but is not bound and uploaded"
                )
            _bind(workflow, bindings.first_frame, uploaded_first_frame)
        if request.last_frame is not None:
            if bindings.last_frame is None or not uploaded_last_frame:
                raise WorkflowBindingError(
                    "last_frame was supplied but is not bound and uploaded"
                )
            _bind(workflow, bindings.last_frame, uploaded_last_frame)

    return workflow


class ComfyUIClient:
    """Small standard-library client for ComfyUI's local HTTP routes."""

    def __init__(self, connection: ComfyUIConnection | None = None):
        self.connection = connection or ComfyUIConnection()

    def _url(self, path: str, query: dict[str, str] | None = None) -> str:
        route = f"{self.connection.api_prefix}/{path.lstrip('/')}"
        url = f"{self.connection.base_url}{route}"
        return f"{url}?{urllib.parse.urlencode(query)}" if query else url

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> bytes:
        request = urllib.request.Request(
            self._url(path, query),
            data=data,
            headers=headers or {},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.connection.request_timeout_seconds
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ComfyUIError(
                f"ComfyUI {method} {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ComfyUIError(
                f"cannot reach ComfyUI at {self.connection.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ComfyUIError(
                f"ComfyUI request to {path} exceeded "
                f"{self.connection.request_timeout_seconds:.1f}s timeout"
            ) from exc

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        raw = self._request(method, path, data=data, headers=headers)
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            raise ComfyUIError(f"ComfyUI returned invalid JSON for {path}") from exc
        if not isinstance(value, dict):
            raise ComfyUIError(f"ComfyUI returned non-object JSON for {path}")
        return value

    def system_stats(self) -> dict[str, Any]:
        return self._json_request("GET", "/system_stats")

    def queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        response = self._json_request(
            "POST", "/prompt", {"prompt": workflow, "client_id": client_id}
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            detail = response.get("error") or response.get("node_errors") or response
            raise ComfyUIError(f"ComfyUI rejected workflow: {detail}")
        return str(prompt_id)

    def wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        started = time.monotonic()
        while time.monotonic() - started < self.connection.job_timeout_seconds:
            history = self._json_request("GET", f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status")
                status_text = (
                    status.get("status_str", "")
                    if isinstance(status, dict)
                    else str(status or "")
                ).lower()
                if status_text in {"error", "failed"}:
                    raise ComfyUIError(
                        f"ComfyUI workflow {prompt_id} failed: {status}"
                    )
                if isinstance(entry.get("outputs"), dict):
                    return entry
            time.sleep(self.connection.poll_interval_seconds)
        raise ComfyUIError(
            f"ComfyUI workflow {prompt_id} exceeded "
            f"{self.connection.job_timeout_seconds:.1f}s timeout"
        )

    def download_output(
        self, filename: str, subfolder: str, output_type: str
    ) -> bytes:
        return self._request(
            "GET",
            "/view",
            query={
                "filename": filename,
                "subfolder": subfolder,
                "type": output_type,
            },
        )

    def upload_image(self, path: Path, remote_name: str) -> str:
        if not path.is_file():
            raise ComfyUIError(f"reference image not found: {path}")
        boundary = f"----video-screener-{uuid4().hex}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = _multipart_body(
            boundary,
            file_field="image",
            filename=remote_name,
            content_type=content_type,
            content=path.read_bytes(),
            fields={"type": "input", "overwrite": "true"},
        )
        raw = self._request(
            "POST",
            "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            response = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ComfyUIError("ComfyUI returned invalid upload JSON") from exc
        if not isinstance(response, dict) or not response.get("name"):
            raise ComfyUIError(f"ComfyUI upload failed: {response}")
        subfolder = str(response.get("subfolder", "")).strip("/")
        if subfolder:
            return str(PurePosixPath(subfolder) / str(response["name"]))
        return str(response["name"])


def _multipart_body(
    boundary: str,
    *,
    file_field: str,
    filename: str,
    content_type: str,
    content: bytes,
    fields: dict[str, str],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
            value.encode("utf-8"),
            b"\r\n",
        ])
    safe_filename = filename.replace('"', "_")
    chunks.extend([
        f"--{boundary}\r\n".encode("ascii"),
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{safe_filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode("ascii"),
    ])
    return b"".join(chunks)


@dataclass(frozen=True)
class _OutputDescriptor:
    filename: str
    subfolder: str
    output_type: str
    node_id: str
    output_key: str
    media_kind: MediaKind


def _kind_for_output(filename: str, output_key: str) -> MediaKind | None:
    suffix = Path(filename).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return MediaKind.VIDEO
    if suffix in IMAGE_EXTENSIONS:
        return MediaKind.IMAGE
    if output_key in {"gifs", "videos"}:
        return MediaKind.VIDEO
    if output_key == "images":
        return MediaKind.IMAGE
    return None


def _extract_outputs(
    history_entry: dict[str, Any], expected_kind: MediaKind
) -> list[_OutputDescriptor]:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    found: list[_OutputDescriptor] = []
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for output_key in OUTPUT_KEYS:
            items = node_output.get(output_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                filename = str(item["filename"])
                kind = _kind_for_output(filename, output_key)
                if kind != expected_kind:
                    continue
                found.append(_OutputDescriptor(
                    filename=filename,
                    subfolder=str(item.get("subfolder", "")),
                    output_type=str(item.get("type", "output")),
                    node_id=str(node_id),
                    output_key=output_key,
                    media_kind=kind,
                ))
    return found


def _safe_remote_name(request_id: str, role: str, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{request_id}_{role}")
    return f"{stem}_{path.name}"


class ComfyUIBackend:
    """Run bound ComfyUI workflows and persist neutral generation results."""

    name = "comfyui"

    def __init__(
        self,
        workflow: dict[str, Any],
        bindings: ComfyUIWorkflowBindings,
        output_dir: str | Path,
        *,
        connection: ComfyUIConnection | None = None,
        client: Any | None = None,
    ):
        self.workflow = validate_comfyui_workflow(workflow)
        self.bindings = bindings
        self.output_dir = Path(output_dir)
        self.client = client or ComfyUIClient(connection)

    @classmethod
    def from_files(
        cls,
        workflow_path: str | Path,
        bindings_path: str | Path,
        output_dir: str | Path,
        *,
        connection: ComfyUIConnection | None = None,
    ) -> "ComfyUIBackend":
        return cls(
            load_comfyui_workflow(workflow_path),
            load_comfyui_bindings(bindings_path),
            output_dir,
            connection=connection,
        )

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> GenerationResult:
        return self._generate(request)

    def generate_video(
        self, request: VideoGenerationRequest
    ) -> GenerationResult:
        return self._generate(request)

    def _generate(
        self, request: ImageGenerationRequest | VideoGenerationRequest
    ) -> GenerationResult:
        started = time.monotonic()
        run_dir = self.output_dir / request.request_id
        manifest_path = run_dir / "generation.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        artifacts: list[GenerationArtifact] = []
        error: str | None = None

        try:
            hardware = self.client.system_stats()
        except ComfyUIError as exc:
            hardware = {}
            warnings.append(f"hardware metadata unavailable: {exc}")

        try:
            uploaded_references = self._upload_references(request)
            uploaded_first = self._upload_frame(request, "first_frame")
            uploaded_last = self._upload_frame(request, "last_frame")
            client_id = str(uuid4())

            for candidate_index in range(request.candidate_count):
                seed = (request.seed + candidate_index) % (MAX_SEED + 1)
                workflow = render_comfyui_workflow(
                    self.workflow,
                    self.bindings,
                    request,
                    seed=seed,
                    uploaded_references=uploaded_references,
                    uploaded_first_frame=uploaded_first,
                    uploaded_last_frame=uploaded_last,
                )
                prompt_id = self.client.queue_prompt(workflow, client_id)
                history = self.client.wait_for_history(prompt_id)
                descriptors = _extract_outputs(history, request.kind)
                if not descriptors:
                    raise ComfyUIError(
                        f"workflow {prompt_id} completed without {request.kind.value} output"
                    )
                artifacts.extend(self._save_outputs(
                    descriptors,
                    run_dir,
                    candidate_index=candidate_index,
                    seed=seed,
                    prompt_id=prompt_id,
                ))
        except (ComfyUIError, OSError, ValueError) as exc:
            error = str(exc)

        status = (
            GenerationStatus.SUCCEEDED if error is None else GenerationStatus.FAILED
        )
        result = GenerationResult(
            request=request,
            status=status,
            backend=self.name,
            runtime_seconds=round(time.monotonic() - started, 3),
            hardware=hardware,
            artifacts=artifacts,
            warnings=warnings,
            error=error,
            manifest_path=manifest_path,
        )
        write_generation_result(result, manifest_path)
        return result

    def _upload_references(
        self, request: ImageGenerationRequest | VideoGenerationRequest
    ) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for reference in request.references:
            if reference.role not in self.bindings.references:
                raise WorkflowBindingError(
                    f"workflow has no binding for reference role {reference.role!r}"
                )
            uploaded[reference.role] = self.client.upload_image(
                reference.path,
                _safe_remote_name(request.request_id, reference.role, reference.path),
            )
        return uploaded

    def _upload_frame(
        self,
        request: ImageGenerationRequest | VideoGenerationRequest,
        field_name: str,
    ) -> str | None:
        if not isinstance(request, VideoGenerationRequest):
            return None
        path = getattr(request, field_name)
        if path is None:
            return None
        binding = getattr(self.bindings, field_name)
        if binding is None:
            raise WorkflowBindingError(
                f"{field_name} was supplied but the workflow has no binding"
            )
        return self.client.upload_image(
            path,
            _safe_remote_name(request.request_id, field_name, path),
        )

    def _save_outputs(
        self,
        descriptors: list[_OutputDescriptor],
        run_dir: Path,
        *,
        candidate_index: int,
        seed: int,
        prompt_id: str,
    ) -> list[GenerationArtifact]:
        candidate_dir = run_dir / f"candidate_{candidate_index:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[GenerationArtifact] = []
        for descriptor in descriptors:
            data = self.client.download_output(
                descriptor.filename,
                descriptor.subfolder,
                descriptor.output_type,
            )
            filename = Path(descriptor.filename).name
            if not filename:
                raise ComfyUIError("ComfyUI returned an empty output filename")
            destination = candidate_dir / filename
            if destination.exists():
                destination = candidate_dir / f"{descriptor.node_id}_{filename}"
            destination.write_bytes(data)
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            artifacts.append(GenerationArtifact(
                path=destination.resolve(),
                media_kind=descriptor.media_kind,
                mime_type=mime_type,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                candidate_index=candidate_index,
                seed=seed,
                source_prompt_id=prompt_id,
                source_node_id=descriptor.node_id,
                metadata={
                    "server_filename": descriptor.filename,
                    "server_subfolder": descriptor.subfolder,
                    "server_type": descriptor.output_type,
                    "output_key": descriptor.output_key,
                },
            ))
        return artifacts
