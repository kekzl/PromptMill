"""Model repository port (interface)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from promptmill.domain.entities.model import Model

if TYPE_CHECKING:
    from collections.abc import Callable


class ModelRepositoryPort(ABC):
    """Port for model storage and retrieval operations.

    This defines the interface for managing model files,
    including downloading from remote sources and local storage.
    """

    @abstractmethod
    def get_model_path(self, model: Model) -> Path | None:
        """Get local path if model exists.

        Args:
            model: The model to locate.

        Returns:
            Path to local model file, or None if not downloaded.
        """
        ...

    @abstractmethod
    def download_model(
        self,
        model: Model,
        progress_callback: "Callable[[float], None] | None" = None,
    ) -> Path:
        """Download model and return local path.

        Args:
            model: The model to download.
            progress_callback: Optional callback for progress updates (0.0-1.0).

        Returns:
            Path to the downloaded model file.

        Raises:
            RuntimeError: If download fails.
            OSError: If there's insufficient disk space.
        """
        ...

    @abstractmethod
    def delete_model(self, model: Model) -> bool:
        """Delete model from local storage.

        Args:
            model: The model to delete.

        Returns:
            True if model was deleted, False if it didn't exist.

        Raises:
            OSError: If deletion fails due to permissions or other issues.
        """
        ...

    @abstractmethod
    def get_disk_usage(self) -> int:
        """Get total disk usage of all models in bytes.

        Returns:
            Total size in bytes of all model files.
        """
        ...

    @abstractmethod
    def get_available_space(self) -> int:
        """Get available disk space in bytes.

        Returns:
            Available space in bytes on the models partition.
        """
        ...


# Type alias for progress callback
