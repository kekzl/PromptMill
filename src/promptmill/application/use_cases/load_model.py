"""Load model use case."""

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from promptmill.domain.entities.model import Model
from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadModelUseCase:
    """Use case for loading a model into the LLM runtime.

    This use case handles:
    1. Checking if model is already loaded
    2. Downloading model if not present locally
    3. Loading the model into the LLM runtime
    4. Thread-safe model switching
    """

    llm: LLMPort
    model_repository: ModelRepositoryPort
    lock: RLock

    def execute(
        self,
        model: Model,
        models_dir: Path,
        n_gpu_layers_override: int | None = None,
    ) -> None:
        """Execute the model loading use case.

        Args:
            model: The model configuration to load.
            models_dir: Directory where models are stored.
            n_gpu_layers_override: GPU offload to use instead of the model's own
                default. None keeps ``model.n_gpu_layers``.

        Raises:
            ModelDownloadError: If download fails.
            ModelLoadError: If loading fails.
        """
        n_gpu_layers = (
            model.n_gpu_layers if n_gpu_layers_override is None else n_gpu_layers_override
        )

        with self.lock:
            # Already loaded means same file AND same GPU split; a different
            # split is a different runtime configuration and needs a reload.
            current_path = self.llm.get_loaded_model_path()
            expected_path = str(models_dir / model.filename)

            if current_path == expected_path and self.llm.get_loaded_gpu_layers() == n_gpu_layers:
                logger.info(f"Model already loaded: {model.name}")
                return

            # Get or download the model
            model_path = self.model_repository.get_model_path(model)
            if model_path is None:
                logger.info(f"Model not found locally, downloading: {model.name}")
                model_path = self.model_repository.download_model(model)

            # Unload current model if any
            if self.llm.is_loaded():
                logger.info("Unloading current model before loading new one")
                self.llm.unload()

            # Load the new model
            logger.info(f"Loading model: {model.name} (n_gpu_layers={n_gpu_layers})")
            self.llm.load(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                context_length=model.context_length,
                chat_format=model.chat_format,
            )
            logger.info(f"Model loaded successfully: {model.name}")
