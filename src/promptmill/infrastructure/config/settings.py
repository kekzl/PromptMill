"""Application settings configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables.

    All settings have sensible defaults and can be overridden via
    environment variables.
    """

    # Server configuration
    host: str = field(
        default_factory=lambda: os.environ.get("SERVER_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("SERVER_PORT", "7610"))
    )

    # Model storage
    models_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MODELS_DIR", "/app/models")
        )
    )

    # LLM configuration
    default_batch_size: int = 512
    default_chat_format: str = "llama-3"
    gpu_detection_timeout: int = 5

    # Input validation limits
    max_prompt_length: int = 10000
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    min_tokens: int = 100
    max_tokens: int = 2000

    # Auto-unload configuration
    unload_delay_seconds: int = 10

    @classmethod
    def from_environment(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings instance with values from environment.
        """
        return cls()

    def ensure_models_dir(self) -> None:
        """Ensure the models directory exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
