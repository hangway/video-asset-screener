"""ComfyUI workflow binding and local persistence tests."""

from __future__ import annotations

import copy
import json
import urllib.parse

import pytest

from video_screener.generation import (
    GenerationStatus,
    ImageGenerationBackend,
    ImageGenerationRequest,
    ModelSpec,
    ReferenceAsset,
    VideoGenerationBackend,
    VideoGenerationRequest,
    WorkflowSpec,
)
from video_screener.generation.comfyui import (
    ComfyUIBackend,
    ComfyUIClient,
    ComfyUIConnection,
    ComfyUIError,
    ComfyUIWorkflowBindings,
    WorkflowBindingError,
    render_comfyui_workflow,
    validate_comfyui_workflow,
)


def _workflow():
    return {
        "1": {"class_type": "Text", "inputs": {"text": "old prompt"}},
        "2": {"class_type": "Text", "inputs": {"text": "old negative"}},
        "3": {"class_type": "Sampler", "inputs": {"seed": 0}},
        "4": {
            "class_type": "Latent",
            "inputs": {"width": 512, "height": 512},
        },
        "5": {
            "class_type": "VideoSettings",
            "inputs": {"fps": 8.0, "frames": 16},
        },
        "6": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "7": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "8": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "9": {"class_type": "Save", "inputs": {"value": ["3", 0]}},
    }


def _bindings(**overrides):
    raw = {
        "prompt": {"node_id": "1", "input_name": "text"},
        "negative_prompt": {"node_id": "2", "input_name": "text"},
        "seed": {"node_id": "3", "input_name": "seed"},
        "width": {"node_id": "4", "input_name": "width"},
        "height": {"node_id": "4", "input_name": "height"},
        "fps": {"node_id": "5", "input_name": "fps"},
        "frame_count": {"node_id": "5", "input_name": "frames"},
        "first_frame": {"node_id": "6", "input_name": "image"},
        "last_frame": {"node_id": "7", "input_name": "image"},
        "references": {
            "character": {"node_id": "8", "input_name": "image"}
        },
    }
    raw.update(overrides)
    return ComfyUIWorkflowBindings.model_validate(raw)


def _model():
    return ModelSpec(
        model_id="local/test-model",
        revision="test-revision",
        license="Apache-2.0",
    )


def _workflow_spec():
    return WorkflowSpec(workflow_id="test-workflow", version="1")


class FakeComfyUIClient:
    def __init__(self, kind="image", *, fail=False, unsafe_filename=False):
        self.kind = kind
        self.fail = fail
        self.unsafe_filename = unsafe_filename
        self.workflows = []
        self.uploads = []

    def system_stats(self):
        return {
            "system": {"comfyui_version": "test"},
            "devices": [{"name": "GPU"}],
        }

    def upload_image(self, path, remote_name):
        self.uploads.append((path, remote_name))
        return f"inputs/{remote_name}"

    def queue_prompt(self, workflow, client_id):
        if self.fail:
            raise ComfyUIError("synthetic queue failure")
        self.workflows.append(copy.deepcopy(workflow))
        return f"prompt-{len(self.workflows) - 1}"

    def wait_for_history(self, prompt_id):
        index = int(prompt_id.rsplit("-", 1)[1])
        if self.kind == "video":
            key = "videos"
            filename = f"candidate-{index}.mp4"
        else:
            key = "images"
            filename = f"candidate-{index}.png"
        if self.unsafe_filename:
            filename = f"../../{filename}"
        return {
            "outputs": {
                "9": {
                    key: [{
                        "filename": filename,
                        "subfolder": "generated",
                        "type": "output",
                    }]
                }
            }
        }

    def download_output(self, filename, subfolder, output_type):
        return f"{filename}|{subfolder}|{output_type}".encode("utf-8")


class _HTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_client_uses_documented_local_http_routes(monkeypatch, tmp_path):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        parsed = urllib.parse.urlparse(request.full_url)
        if parsed.path == "/system_stats":
            payload = {"devices": [{"name": "GPU"}]}
        elif parsed.path == "/prompt":
            queued = json.loads(request.data.decode("utf-8"))
            assert queued["prompt"]["1"]["class_type"] == "Text"
            payload = {"prompt_id": "prompt-1"}
        elif parsed.path == "/history/prompt-1":
            payload = {"prompt-1": {"outputs": {"9": {"images": []}}}}
        elif parsed.path == "/upload/image":
            assert b'name="type"' in request.data
            assert b'name="image"' in request.data
            payload = {"name": "uploaded.png", "subfolder": "inputs"}
        elif parsed.path == "/view":
            query = urllib.parse.parse_qs(parsed.query)
            assert query["filename"] == ["result.png"]
            return _HTTPResponse(b"image-bytes")
        else:  # pragma: no cover - makes unexpected routes obvious
            raise AssertionError(request.full_url)
        return _HTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ComfyUIClient(ComfyUIConnection(request_timeout_seconds=12.0))
    upload = tmp_path / "upload.png"
    upload.write_bytes(b"upload")

    assert client.system_stats()["devices"][0]["name"] == "GPU"
    assert client.queue_prompt(_workflow(), "client-1") == "prompt-1"
    assert "outputs" in client.wait_for_history("prompt-1")
    assert client.upload_image(upload, "remote.png") == "inputs/uploaded.png"
    assert client.download_output("result.png", "", "output") == b"image-bytes"
    assert all(timeout == 12.0 for _, timeout in requests)


def test_api_workflow_validation_rejects_ui_format():
    with pytest.raises(ValueError, match="API format"):
        validate_comfyui_workflow({"nodes": []})


def test_render_workflow_rejects_unknown_binding_node():
    bindings = _bindings(prompt={"node_id": "missing", "input_name": "text"})
    request = ImageGenerationRequest(
        model=_model(),
        workflow=_workflow_spec(),
        prompt="hello",
        seed=1,
    )
    with pytest.raises(WorkflowBindingError, match="does not exist"):
        render_comfyui_workflow(_workflow(), bindings, request, seed=1)


def test_empty_negative_prompt_clears_workflow_default():
    request = ImageGenerationRequest(
        model=_model(),
        workflow=_workflow_spec(),
        prompt="hello",
        negative_prompt="",
        seed=1,
    )

    rendered = render_comfyui_workflow(
        _workflow(), _bindings(), request, seed=1
    )

    assert rendered["2"]["inputs"]["text"] == ""


def test_image_backend_generates_reproducible_candidates_and_manifest(tmp_path):
    reference = tmp_path / "character.png"
    reference.write_bytes(b"reference")
    client = FakeComfyUIClient(unsafe_filename=True)
    backend = ComfyUIBackend(
        _workflow(), _bindings(), tmp_path / "runs", client=client
    )
    request = ImageGenerationRequest(
        request_id="image-job",
        model=_model(),
        workflow=_workflow_spec(),
        prompt="consistent character portrait",
        negative_prompt="text artifacts",
        seed=40,
        candidate_count=2,
        width=768,
        height=512,
        references=(ReferenceAsset(path=reference, role="character"),),
    )

    result = backend.generate_image(request)

    assert isinstance(backend, ImageGenerationBackend)
    assert result.status == GenerationStatus.SUCCEEDED
    assert [artifact.seed for artifact in result.artifacts] == [40, 41]
    assert [artifact.candidate_index for artifact in result.artifacts] == [0, 1]
    assert all(artifact.path.is_file() for artifact in result.artifacts)
    assert all(
        artifact.path.parent.name.startswith("candidate_")
        for artifact in result.artifacts
    )
    assert all(".." not in artifact.path.name for artifact in result.artifacts)
    assert client.workflows[0]["1"]["inputs"]["text"] == request.prompt
    assert client.workflows[0]["2"]["inputs"]["text"] == request.negative_prompt
    assert client.workflows[0]["3"]["inputs"]["seed"] == 40
    assert client.workflows[1]["3"]["inputs"]["seed"] == 41
    assert client.workflows[0]["4"]["inputs"] == {
        "width": 768,
        "height": 512,
    }
    assert client.workflows[0]["8"]["inputs"]["image"].startswith("inputs/")
    saved = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "succeeded"
    assert saved["request"]["model"]["model_id"] == "local/test-model"
    assert saved["hardware"]["devices"][0]["name"] == "GPU"


def test_video_backend_uploads_keyframes_and_binds_frame_count(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    client = FakeComfyUIClient(kind="video")
    backend = ComfyUIBackend(
        _workflow(), _bindings(), tmp_path / "runs", client=client
    )
    request = VideoGenerationRequest(
        request_id="video-job",
        model=_model(),
        workflow=_workflow_spec(),
        prompt="camera slowly moves forward",
        seed=7,
        width=1280,
        height=720,
        duration_seconds=2.5,
        fps=16,
        first_frame=first,
        last_frame=last,
    )

    result = backend.generate_video(request)

    assert isinstance(backend, VideoGenerationBackend)
    assert result.status == GenerationStatus.SUCCEEDED
    assert result.artifacts[0].path.suffix == ".mp4"
    assert client.workflows[0]["5"]["inputs"] == {
        "fps": 16.0,
        "frames": 40,
    }
    assert client.workflows[0]["6"]["inputs"]["image"].startswith("inputs/")
    assert client.workflows[0]["7"]["inputs"]["image"].startswith("inputs/")
    assert len(client.uploads) == 2


def test_backend_writes_failed_manifest_for_expected_runtime_error(tmp_path):
    client = FakeComfyUIClient(fail=True)
    backend = ComfyUIBackend(
        _workflow(), _bindings(), tmp_path / "runs", client=client
    )
    request = ImageGenerationRequest(
        request_id="failed-job",
        model=_model(),
        workflow=_workflow_spec(),
        prompt="test",
        seed=1,
    )

    result = backend.generate_image(request)

    assert result.status == GenerationStatus.FAILED
    assert result.error == "synthetic queue failure"
    assert result.artifacts == []
    saved = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["error"] == "synthetic queue failure"
