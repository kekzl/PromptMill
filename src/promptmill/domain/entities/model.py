"""Model entity representing an LLM configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Model:
    """Domain entity representing an LLM model configuration.

    This is an immutable value object that describes a model's configuration
    without managing its lifecycle.

    Attributes:
        key: Unique identifier for this model configuration (e.g., "8gb_vram").
        name: Human-readable display name.
        repo_id: HuggingFace repository ID.
        filename: Model filename within the repository.
        context_length: Maximum context window size.
        n_gpu_layers: Number of layers to offload to GPU (-1 for all).
        description: Human-readable description of the model.
        vram_required: Approximate VRAM requirement string.
        revision: Git revision (commit hash) for reproducible downloads.
    """

    key: str
    name: str
    repo_id: str
    filename: str
    context_length: int
    n_gpu_layers: int
    description: str
    vram_required: str
    revision: str | None = None

    def get_local_path(self, models_dir: Path) -> Path:
        """Get the expected local file path for this model.

        Args:
            models_dir: Base directory for model storage.

        Returns:
            Path where the model file would be stored.
        """
        return models_dir / self.filename

    def is_downloaded(self, models_dir: Path) -> bool:
        """Check if model file exists locally.

        Args:
            models_dir: Base directory for model storage.

        Returns:
            True if model file exists.
        """
        return self.get_local_path(models_dir).exists()
