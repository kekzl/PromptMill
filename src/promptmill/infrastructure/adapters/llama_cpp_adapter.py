"""LlamaCpp adapter implementing LLMPort."""

import gc
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing import TypeVar

    def override(func: TypeVar("F")) -> TypeVar("F"):  # type: ignore[misc]
        return func  # type: ignore[return-value]

from promptmill.domain.exceptions import ModelLoadError, ModelNotLoadedError
from promptmill.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class LlamaCppAdapter(LLMPort):
    """Adapter for llama-cpp-python LLM inference.

    This adapter implements the LLMPort interface using llama-cpp-python
    for local LLM inference.
    """

    __slots__ = ("_batch_size", "_chat_format", "_llm", "_model_path")

    def __init__(
        self,
        chat_format: str = "llama-3",
        batch_size: int = 512,
    ) -> None:
        """Initialize the adapter.

        Args:
            chat_format: Chat template format to use.
            batch_size: Batch size for inference.
        """
        self._llm: Any = None  # Llama instance, Any to avoid import at module level
        self._model_path: str | None = None
        self._chat_format = chat_format
        self._batch_size = batch_size

    @override
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """Generate text with streaming output.

        Args:
            system_prompt: The system prompt defining the AI's role.
            user_prompt: The user's input to respond to.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they are generated.

        Raises:
            ModelNotLoadedError: If no model is loaded.
        """
        if self._llm is None:
            raise ModelNotLoadedError()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            f"Generating with temp={temperature}, max_tokens={max_tokens}"
        )

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    @override
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if a model is loaded and ready for inference.
        """
        return self._llm is not None

    @override
    def get_loaded_model_path(self) -> str | None:
        """Get the path of the currently loaded model.

        Returns:
            Path string if a model is loaded, None otherwise.
        """
        return self._model_path

    @override
    def load(
        self,
        model_path: str,
        n_gpu_layers: int,
        context_length: int,
    ) -> None:
        """Load a model for inference.

        Args:
            model_path: Path to the model file.
            n_gpu_layers: Number of layers to offload to GPU (-1 for all).
            context_length: Maximum context window size.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ModelLoadError: If model loading fails.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Unload any existing model first
        if self._llm is not None:
            self.unload()

        logger.info(f"Loading model: {model_path}")
        logger.info(
            f"Config: n_gpu_layers={n_gpu_layers}, n_ctx={context_length}"
        )

        try:
            # Lazy import to avoid loading llama_cpp until needed
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                n_batch=self._batch_size,
                chat_format=self._chat_format,
                verbose=False,
            )
            self._model_path = model_path
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._llm = None
            self._model_path = None
            raise ModelLoadError(model_path, str(e)) from e

    @override
    def unload(self) -> None:
        """Unload the current model and free resources."""
        if self._llm is not None:
            logger.info(f"Unloading model: {self._model_path}")
            del self._llm
            self._llm = None
            self._model_path = None
            gc.collect()
            logger.info("Model unloaded, memory freed")
