"""Prompt generation service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock, Timer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from promptmill.application.use_cases.generate_prompt import GeneratePromptUseCase
    from promptmill.application.use_cases.load_model import LoadModelUseCase
    from promptmill.application.use_cases.unload_model import UnloadModelUseCase
    from promptmill.domain.entities.model import Model
    from promptmill.domain.value_objects.prompt_request import PromptGenerationRequest

logger = logging.getLogger(__name__)


@dataclass
class PromptService:
    """High-level service for prompt generation with auto-unload.

    This service coordinates:
    - Model loading when needed
    - Prompt generation
    - Automatic model unloading after inactivity
    """

    generate_prompt_use_case: GeneratePromptUseCase
    load_model_use_case: LoadModelUseCase
    unload_model_use_case: UnloadModelUseCase
    models_dir: Path
    unload_delay_seconds: int = 10

    _current_model: Model | None = None
    _unload_timer: Timer | None = None
    _timer_lock: RLock | None = None

    def __post_init__(self) -> None:
        """Initialize timer lock."""
        self._timer_lock = RLock()

    def generate(
        self,
        request: PromptGenerationRequest,
        model: Model,
    ) -> Iterator[str]:
        """Generate a prompt with automatic model management.

        This method:
        1. Loads the model if not already loaded
        2. Generates the prompt
        3. Schedules auto-unload timer

        Args:
            request: The prompt generation request.
            model: The model to use for generation.

        Yields:
            Text chunks as they are generated.
        """
        # Cancel any pending unload
        self._cancel_unload_timer()

        # Load model if needed
        self.load_model_use_case.execute(model, self.models_dir)
        self._current_model = model

        try:
            # Generate prompt
            yield from self.generate_prompt_use_case.execute(request)
        finally:
            # Schedule unload after generation
            self._schedule_unload()

    def _schedule_unload(self) -> None:
        """Schedule automatic model unload after delay."""
        with self._timer_lock:
            self._cancel_unload_timer()
            self._unload_timer = Timer(
                self.unload_delay_seconds,
                self._auto_unload,
            )
            self._unload_timer.daemon = True
            self._unload_timer.start()
            logger.debug(f"Scheduled model unload in {self.unload_delay_seconds} seconds")

    def _cancel_unload_timer(self) -> None:
        """Cancel any pending unload timer."""
        with self._timer_lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None

    def _auto_unload(self) -> None:
        """Automatically unload model after timeout."""
        with self._timer_lock:
            self._unload_timer = None

        logger.info("Auto-unloading model due to inactivity")
        self.unload_model_use_case.execute()
        self._current_model = None

    def shutdown(self) -> None:
        """Clean shutdown - cancel timers and unload model."""
        self._cancel_unload_timer()
        self.unload_model_use_case.execute()
        self._current_model = None

    @property
    def current_model(self) -> Model | None:
        """Get currently loaded model."""
        return self._current_model
