"""Integration tests for the REST API."""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("gradio")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from promptmill.domain.entities.model import Model
from promptmill.domain.exceptions import GenerationError
from promptmill.infrastructure.adapters.role_repository_adapter import RoleRepositoryAdapter
from promptmill.presentation.api import create_api_router

TEST_MODEL = Model(
    key="cpu_only",
    name="CPU Only (2-4GB RAM)",
    repo_id="test/repo",
    filename="test.gguf",
    context_length=4096,
    n_gpu_layers=0,
    description="Test tier",
    vram_required="~1.5GB",
)


def _failing_after(n: int) -> Callable[..., Iterator[str]]:
    """Build a generate() stand-in that yields n chunks, then fails."""

    def generate(*_a: object, **_k: object) -> Iterator[str]:
        yield from ["chunk"] * n
        raise GenerationError("ctx full")

    return generate


@pytest.fixture
def prompt_service() -> MagicMock:
    """Prompt service that streams a fixed answer."""
    service = MagicMock()
    service.generate.side_effect = lambda *_a, **_k: iter(
        ["a cinematic ", "shot of a ", "lighthouse"]
    )
    return service


@pytest.fixture
def model_service() -> MagicMock:
    """Model service exposing a single tier."""
    service = MagicMock()
    service.get_available_models.return_value = [TEST_MODEL]
    service.get_model_by_name.side_effect = lambda n: TEST_MODEL if n == TEST_MODEL.name else None
    service.select_optimal_model.return_value = (TEST_MODEL, None)
    service.is_model_downloaded.return_value = False
    return service


@pytest.fixture
def client(prompt_service: MagicMock, model_service: MagicMock) -> TestClient:
    """A TestClient over the API router alone, without Gradio mounted."""
    health_service = MagicMock()
    health_service.get_status.return_value = {"status": "healthy", "roles_count": 146}

    app = FastAPI()
    app.include_router(
        create_api_router(
            prompt_service=prompt_service,
            model_service=model_service,
            health_service=health_service,
            role_repository=RoleRepositoryAdapter(),
            default_model=TEST_MODEL,
        )
    )
    return TestClient(app)


@pytest.fixture
def target(client: TestClient) -> str:
    """A display name that is guaranteed to exist."""
    return client.get("/api/targets", params={"category": "Video"}).json()[0]["display_name"]


class TestHealth:
    """Tests for /health."""

    def test_health_returns_status(self, client: TestClient) -> None:
        """The health endpoint answers with the service status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestTargets:
    """Tests for /api/targets."""

    def test_lists_all_targets(self, client: TestClient) -> None:
        """Every role is exposed."""
        response = client.get("/api/targets")
        assert response.status_code == 200
        assert len(response.json()) == RoleRepositoryAdapter().count()

    def test_filters_by_category(self, client: TestClient) -> None:
        """A category filter narrows the list."""
        body = client.get("/api/targets", params={"category": "Audio"}).json()
        assert body
        assert {t["category"] for t in body} == {"Audio"}

    def test_category_is_case_insensitive(self, client: TestClient) -> None:
        """Callers should not have to match casing exactly."""
        assert client.get("/api/targets", params={"category": "audio"}).status_code == 200

    def test_unknown_category_is_404(self, client: TestClient) -> None:
        """An unknown category is an error, not a silent fallback."""
        response = client.get("/api/targets", params={"category": "Sculpture"})
        assert response.status_code == 404

    def test_target_shape(self, client: TestClient) -> None:
        """Each entry carries the fields a client needs."""
        entry = client.get("/api/targets").json()[0]
        assert set(entry) == {"name", "category", "display_name", "description"}


class TestModels:
    """Tests for /api/models."""

    def test_lists_models(self, client: TestClient) -> None:
        """Model tiers are exposed with their download state."""
        body = client.get("/api/models").json()
        assert len(body) == 1
        assert body[0]["name"] == TEST_MODEL.name
        assert body[0]["downloaded"] is False
        assert body[0]["context_length"] == 4096


class TestGenerate:
    """Tests for /api/generate."""

    def test_generates_prompt(self, client: TestClient, target: str) -> None:
        """A valid request returns the joined prompt and its counts."""
        response = client.post("/api/generate", json={"input": "a lighthouse", "target": target})
        assert response.status_code == 200
        body = response.json()
        assert body["prompt"] == "a cinematic shot of a lighthouse"
        assert body["target"] == target
        assert body["model"] == TEST_MODEL.name
        assert body["characters"] == len(body["prompt"])
        assert body["words"] == 6

    def test_defaults_to_auto_selected_model(
        self, client: TestClient, target: str, model_service: MagicMock
    ) -> None:
        """Omitting the model uses the tier detected at startup, without re-running nvidia-smi."""
        response = client.post("/api/generate", json={"input": "idea", "target": target})
        assert response.json()["model"] == TEST_MODEL.name
        model_service.select_optimal_model.assert_not_called()

    def test_generation_failure_is_503(
        self, client: TestClient, target: str, prompt_service: MagicMock
    ) -> None:
        """A runtime failure maps to 503 with detail, not a bare 500."""
        prompt_service.generate.side_effect = _failing_after(0)
        response = client.post("/api/generate", json={"input": "idea", "target": target})
        assert response.status_code == 503
        assert "ctx full" in response.json()["detail"]

    def test_explicit_model_is_used(self, client: TestClient, target: str) -> None:
        """A named model is honoured."""
        response = client.post(
            "/api/generate",
            json={"input": "idea", "target": target, "model": TEST_MODEL.name},
        )
        assert response.status_code == 200

    def test_unknown_target_is_404(self, client: TestClient) -> None:
        """An unknown target is rejected before any model is loaded."""
        response = client.post("/api/generate", json={"input": "idea", "target": "[Video] Nope"})
        assert response.status_code == 404

    def test_unknown_model_is_404(self, client: TestClient, target: str) -> None:
        """An unknown model name is rejected."""
        response = client.post(
            "/api/generate", json={"input": "idea", "target": target, "model": "nope"}
        )
        assert response.status_code == 404

    def test_empty_input_is_422(self, client: TestClient, target: str) -> None:
        """Blank input fails validation."""
        response = client.post("/api/generate", json={"input": "", "target": target})
        assert response.status_code == 422

    def test_out_of_range_temperature_is_422(self, client: TestClient, target: str) -> None:
        """Temperature bounds match the domain value object."""
        response = client.post(
            "/api/generate", json={"input": "idea", "target": target, "temperature": 9.0}
        )
        assert response.status_code == 422

    def test_out_of_range_tokens_is_422(self, client: TestClient, target: str) -> None:
        """Token bounds match the domain value object."""
        response = client.post(
            "/api/generate", json={"input": "idea", "target": target, "max_tokens": 99999}
        )
        assert response.status_code == 422

    def test_generation_parameters_reach_the_service(
        self, client: TestClient, target: str, prompt_service: MagicMock
    ) -> None:
        """Temperature and token limit are passed through, not dropped."""
        client.post(
            "/api/generate",
            json={"input": "idea", "target": target, "temperature": 1.3, "max_tokens": 400},
        )
        request = prompt_service.generate.call_args.args[0]
        assert request.temperature == 1.3
        assert request.max_tokens == 400


class TestGenerateStream:
    """Tests for /api/generate/stream."""

    def test_streams_plain_text(self, client: TestClient, target: str) -> None:
        """The streaming endpoint returns the same text as the JSON one."""
        response = client.post(
            "/api/generate/stream", json={"input": "a lighthouse", "target": target}
        )
        assert response.status_code == 200
        assert response.text == "a cinematic shot of a lighthouse"

    def test_stream_load_failure_is_503(
        self, client: TestClient, target: str, prompt_service: MagicMock
    ) -> None:
        """A failure before the first chunk is a 503, not a 200 with error text."""
        prompt_service.generate.side_effect = _failing_after(0)
        response = client.post("/api/generate/stream", json={"input": "idea", "target": target})
        assert response.status_code == 503
        assert "ctx full" in response.json()["detail"]

    def test_stream_mid_failure_appends_error(
        self, client: TestClient, target: str, prompt_service: MagicMock
    ) -> None:
        """Once streaming has started, a failure is appended to the body."""
        prompt_service.generate.side_effect = _failing_after(1)
        response = client.post("/api/generate/stream", json={"input": "idea", "target": target})
        assert response.status_code == 200
        assert response.text == "chunk\n[error] Generation failed: ctx full"

    def test_stream_rejects_unknown_target(self, client: TestClient) -> None:
        """Validation happens before streaming starts."""
        response = client.post(
            "/api/generate/stream", json={"input": "idea", "target": "[Video] Nope"}
        )
        assert response.status_code == 404


class TestOpenAPI:
    """The schema is the contract third parties read."""

    def test_routes_are_documented(self, client: TestClient) -> None:
        """All API routes appear in the OpenAPI schema."""
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/api/targets", "/api/models", "/api/generate"} <= set(paths)


def test_assets_dir_is_not_required(tmp_path: Path) -> None:
    """The API router works without any asset directory present."""
    assert not (tmp_path / "assets").exists()
