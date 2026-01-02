"""HuggingFace Hub adapter implementing ModelRepositoryPort."""

import contextlib
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing import TypeVar

    def override(func: TypeVar("F")) -> TypeVar("F"):  # type: ignore[misc]
        return func  # type: ignore[return-value]

from promptmill.domain.entities.model import Model
from promptmill.domain.exceptions import ModelDownloadError
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(ModelRepositoryPort):
    """Adapter for HuggingFace Hub model repository.

    This adapter implements the ModelRepositoryPort interface using
    huggingface_hub for downloading models.
    """

    __slots__ = ("_models_dir",)

    def __init__(self, models_dir: Path) -> None:
        """Initialize the adapter.

        Args:
            models_dir: Base directory for model storage.
        """
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    @override
    def get_model_path(self, model: Model) -> Path | None:
        """Get local path if model exists.

        Args:
            model: The model to locate.

        Returns:
            Path to local model file, or None if not downloaded.
        """
        path = self._models_dir / model.filename
        return path if path.exists() else None

    @override
    def download_model(
        self,
        model: Model,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Path:
        """Download model and return local path.

        Args:
            model: The model to download.
            progress_callback: Optional callback for progress updates (0.0-1.0).

        Returns:
            Path to the downloaded model file.

        Raises:
            ModelDownloadError: If download fails.
        """
        logger.info(f"Downloading model: {model.name}")
        logger.info(f"  Repo: {model.repo_id}")
        logger.info(f"  File: {model.filename}")

        try:
            # Lazy import to avoid loading huggingface_hub until needed
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=model.repo_id,
                filename=model.filename,
                local_dir=self._models_dir,
                local_dir_use_symlinks=False,
            )

            result_path = Path(local_path)
            logger.info(f"Model downloaded: {result_path}")

            if progress_callback:
                progress_callback(1.0)

            return result_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise ModelDownloadError(model.name, str(e)) from e

    @override
    def delete_model(self, model: Model) -> bool:
        """Delete model from local storage.

        Args:
            model: The model to delete.

        Returns:
            True if model was deleted, False if it didn't exist.
        """
        path = self._models_dir / model.filename
        if path.exists():
            path.unlink()
            logger.info(f"Model deleted: {path}")
            return True
        return False

    @override
    def get_disk_usage(self) -> int:
        """Get total disk usage of all models in bytes.

        Returns:
            Total size in bytes of all model files.
        """
        total = 0
        for file in self._models_dir.rglob("*"):
            if file.is_file():
                with contextlib.suppress(OSError):
                    total += file.stat().st_size
        return total

    @override
    def get_available_space(self) -> int:
        """Get available disk space in bytes.

        Returns:
            Available space in bytes on the models partition.
        """
        try:
            usage = shutil.disk_usage(self._models_dir)
            return usage.free
        except OSError:
            return 0

    def format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format.

        Args:
            size_bytes: Size in bytes.

        Returns:
            Human-readable size string (e.g., "4.5 GB").
        """
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
