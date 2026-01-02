"""LLM inference port (interface)."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMPort(ABC):
    """Port for LLM inference operations.

    This defines the interface that any LLM adapter must implement.
    The domain layer depends only on this interface, not on concrete
    implementations like llama-cpp-python.
    """

    @abstractmethod
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
            temperature: Sampling temperature (higher = more random).
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they are generated.

        Raises:
            RuntimeError: If no model is loaded.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if a model is loaded and ready for inference.
        """
        ...

    @abstractmethod
    def get_loaded_model_path(self) -> str | None:
        """Get the path of the currently loaded model.

        Returns:
            Path string if a model is loaded, None otherwise.
        """
        ...

    @abstractmethod
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
            RuntimeError: If model loading fails.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the current model and free resources.

        This should be safe to call even if no model is loaded.
        """
        ...
