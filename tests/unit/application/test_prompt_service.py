"""Tests for PromptService model lifecycle and serialization."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from promptmill.application.services.prompt_service import PromptService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from promptmill.domain.entities.model import Model


@pytest.fixture
def service(tmp_path: Path) -> Iterator[PromptService]:
    generate = MagicMock()
    generate.execute.side_effect = lambda _request: iter(["a", "b"])
    svc = PromptService(
        generate_prompt_use_case=generate,
        load_model_use_case=MagicMock(),
        unload_model_use_case=MagicMock(),
        models_dir=tmp_path,
        unload_delay_seconds=3600,
    )
    yield svc
    svc._cancel_unload_timer()


def test_auto_unload_skips_running_generation(service: PromptService, sample_model: Model) -> None:
    gen = service.generate(MagicMock(), sample_model)
    assert next(gen) == "a"

    service._auto_unload()

    service.unload_model_use_case.execute.assert_not_called()
    assert list(gen) == ["b"]


def test_auto_unload_unloads_when_idle(service: PromptService, sample_model: Model) -> None:
    assert list(service.generate(MagicMock(), sample_model)) == ["a", "b"]

    service._auto_unload()

    service.unload_model_use_case.execute.assert_called_once()
    assert service.current_model is None


def test_generations_are_serialized(service: PromptService, sample_model: Model) -> None:
    first = service.generate(MagicMock(), sample_model)
    next(first)

    second_started = threading.Event()

    def run_second() -> None:
        list(service.generate(MagicMock(), sample_model))
        second_started.set()

    thread = threading.Thread(target=run_second, daemon=True)
    thread.start()

    assert not second_started.wait(0.2)
    assert service.load_model_use_case.execute.call_count == 1

    list(first)
    assert second_started.wait(5)
    assert service.load_model_use_case.execute.call_count == 2


def test_failed_load_releases_lock(service: PromptService, sample_model: Model) -> None:
    service.load_model_use_case.execute.side_effect = [RuntimeError("boom"), None]

    with pytest.raises(RuntimeError):
        list(service.generate(MagicMock(), sample_model))

    assert list(service.generate(MagicMock(), sample_model)) == ["a", "b"]
