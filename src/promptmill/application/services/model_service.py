"""Model management service."""

import logging
from dataclasses import dataclass
from pathlib import Path

from promptmill.application.use_cases.delete_model import DeleteModelUseCase
from promptmill.application.use_cases.load_model import LoadModelUseCase
from promptmill.application.use_cases.select_model_by_vram import SelectModelByVRAMUseCase
from promptmill.application.use_cases.unload_model import UnloadModelUseCase
from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort
from promptmill.infrastructure.config.model_configs import (
    get_all_models,
    get_model_by_name,
    get_model_names,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelService:
    """High-level service for model management.

    This service provides a simplified interface for:
    - Listing available models
    - Selecting optimal model for hardware
    - Loading and unloading models
    - Managing downloaded models
    """

    load_model_use_case: LoadModelUseCase
    unload_model_use_case: UnloadModelUseCase
    delete_model_use_case: DeleteModelUseCase
    select_model_use_case: SelectModelByVRAMUseCase
    model_repository: ModelRepositoryPort
    models_dir: Path

    def get_available_models(self) -> list[Model]:
        """Get all available model configurations.

        Returns:
            List of all Model configurations.
        """
        return get_all_models()

    def get_model_names(self) -> list[str]:
        """Get display names of all models.

        Returns:
            List of model display names.
        """
        return get_model_names()

    def get_model_by_name(self, name: str) -> Model | None:
        """Get model by display name.

        Args:
            name: Model display name.

        Returns:
            Model if found, None otherwise.
        """
        return get_model_by_name(name)

    def select_optimal_model(self) -> tuple[Model, GPUInfo | None]:
        """Select the optimal model based on detected hardware.

        Returns:
            Tuple of (selected_model, gpu_info).
        """
        return self.select_model_use_case.execute()

    def load_model(self, model: Model) -> None:
        """Load a model.

        Args:
            model: Model to load.
        """
        self.load_model_use_case.execute(model, self.models_dir)

    def unload_model(self) -> bool:
        """Unload current model.

        Returns:
            True if model was unloaded.
        """
        return self.unload_model_use_case.execute()

    def delete_model(self, model: Model) -> bool:
        """Delete a downloaded model.

        Args:
            model: Model to delete.

        Returns:
            True if model was deleted.
        """
        return self.delete_model_use_case.execute(model, self.models_dir)

    def is_model_downloaded(self, model: Model) -> bool:
        """Check if a model is downloaded.

        Args:
            model: Model to check.

        Returns:
            True if model file exists locally.
        """
        return self.model_repository.get_model_path(model) is not None

    def get_disk_usage(self) -> int:
        """Get disk usage of all models.

        Returns:
            Total disk usage in bytes.
        """
        return self.model_repository.get_disk_usage()

    def get_disk_usage_formatted(self) -> str:
        """Get formatted disk usage string.

        Returns:
            Human-readable disk usage string.
        """
        usage = self.get_disk_usage()
        return self._format_size(usage)

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
