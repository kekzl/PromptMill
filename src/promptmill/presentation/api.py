"""REST API for PromptMill.

Everything the UI can do with a prompt is reachable over HTTP, so PromptMill is
usable from a script or another service without driving Gradio.
"""

import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from promptmill.application.services.health_service import HealthService
from promptmill.application.services.model_service import ModelService
from promptmill.application.services.prompt_service import PromptService
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import RoleCategory
from promptmill.domain.exceptions import PromptMillError
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort
from promptmill.domain.value_objects.prompt_request import (
    MAX_INPUT_LENGTH,
    MAX_TEMPERATURE,
    MAX_TOKENS,
    MIN_TEMPERATURE,
    MIN_TOKENS,
    PromptGenerationRequest,
)
from promptmill.domain.value_objects.prompt_result import PromptGenerationResult

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256


class GenerateRequest(BaseModel):
    """Body of a POST /api/generate call."""

    input: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description="The idea to turn into a prompt.",
        examples=["a lone lighthouse in a storm at dusk"],
    )
    target: str = Field(
        ...,
        description="Target display name, as returned by GET /api/targets.",
        examples=["[Video] Sora 2"],
    )
    model: str | None = Field(
        default=None,
        description="Model display name from GET /api/models. Defaults to the auto-selected model.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        description="Sampling temperature.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=MIN_TOKENS,
        le=MAX_TOKENS,
        description="Maximum tokens to generate.",
    )


class GenerateResponse(BaseModel):
    """Result of a completed generation."""

    prompt: str
    target: str
    model: str
    characters: int
    words: int


class TargetInfo(BaseModel):
    """One prompt target."""

    name: str
    category: str
    display_name: str
    description: str


class ModelInfo(BaseModel):
    """One selectable LLM tier."""

    name: str
    key: str
    description: str
    vram_required: str
    context_length: int
    downloaded: bool


def create_api_router(
    prompt_service: PromptService,
    model_service: ModelService,
    health_service: HealthService,
    role_repository: RoleRepositoryPort,
) -> APIRouter:
    """Build the REST router bound to the application services.

    Args:
        prompt_service: Service used for generation.
        model_service: Service used for model lookup and status.
        health_service: Service backing the health endpoint.
        role_repository: Repository used to list targets.

    Returns:
        Router with the health and /api routes mounted.
    """
    router = APIRouter()

    def _resolve_model(name: str | None) -> Model:
        """Resolve a model name, falling back to the auto-selected tier."""
        if name is None:
            auto_selected, _ = model_service.select_optimal_model()
            return auto_selected

        named = model_service.get_model_by_name(name)
        if named is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
        return named

    def _build_request(body: GenerateRequest) -> PromptGenerationRequest:
        """Turn the HTTP body into a validated domain request."""
        if role_repository.get_by_display_name(body.target) is None:
            raise HTTPException(status_code=404, detail=f"Unknown target: {body.target}")
        try:
            return PromptGenerationRequest(
                user_input=body.input,
                role_display_name=body.target,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @router.get("/health", tags=["status"])
    def health_check() -> JSONResponse:
        """Health check endpoint for container orchestration."""
        status = health_service.get_status()
        return JSONResponse(content=dict(status))

    @router.get("/api/targets", response_model=list[TargetInfo], tags=["catalog"])
    def list_targets(
        category: Annotated[
            str | None,
            Query(description="Filter by category: Video, Image, Audio, 3D, Creative."),
        ] = None,
    ) -> list[TargetInfo]:
        """List available prompt targets."""
        if category is None:
            roles = role_repository.get_all()
        else:
            parsed = RoleCategory.from_string(category)
            if parsed.value.lower() != category.lower():
                raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
            roles = role_repository.get_by_category(parsed)

        return [
            TargetInfo(
                name=role.name,
                category=role.category.value,
                display_name=role.display_name,
                description=role.description,
            )
            for role in roles
        ]

    @router.get("/api/models", response_model=list[ModelInfo], tags=["catalog"])
    def list_models() -> list[ModelInfo]:
        """List the selectable model tiers and whether they are downloaded."""
        return [
            ModelInfo(
                name=model.name,
                key=model.key,
                description=model.description,
                vram_required=model.vram_required,
                context_length=model.context_length,
                downloaded=model_service.is_model_downloaded(model),
            )
            for model in model_service.get_available_models()
        ]

    @router.post("/api/generate", response_model=GenerateResponse, tags=["generate"])
    def generate(body: GenerateRequest) -> GenerateResponse:
        """Generate a prompt and return it once complete."""
        request = _build_request(body)
        model = _resolve_model(body.model)

        try:
            content = "".join(prompt_service.generate(request, model))
        except PromptMillError as e:
            logger.exception("API generation failed")
            raise HTTPException(status_code=503, detail=str(e)) from e

        result = PromptGenerationResult(
            content=content,
            model_used=model.name,
            role_used=body.target,
        )
        return GenerateResponse(
            prompt=result.content,
            target=body.target,
            model=model.name,
            characters=result.char_count,
            words=result.word_count,
        )

    @router.post("/api/generate/stream", tags=["generate"])
    def generate_stream(body: GenerateRequest) -> StreamingResponse:
        """Generate a prompt, streaming plain-text chunks as they arrive."""
        request = _build_request(body)
        model = _resolve_model(body.model)

        def chunks() -> Iterator[str]:
            try:
                yield from prompt_service.generate(request, model)
            except PromptMillError as e:
                logger.exception("API stream failed")
                yield f"\n[error] {e}"

        return StreamingResponse(chunks(), media_type="text/plain; charset=utf-8")

    return router
