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

    def execute(self, model: Model, models_dir: Path) -> None:
        """Execute the model loading use case.

        Args:
            model: The model configuration to load.
            models_dir: Directory where models are stored.

        Raises:
            ModelDownloadError: If download fails.
            ModelLoadError: If loading fails.
        """
        with self.lock:
            # Check if this model is already loaded
            current_path = self.llm.get_loaded_model_path()
            expected_path = str(models_dir / model.filename)

            if current_path == expected_path:
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
            logger.info(f"Loading model: {model.name}")
            self.llm.load(
                model_path=str(model_path),
                n_gpu_layers=model.n_gpu_layers,
                context_length=model.context_length,
            )
            logger.info(f"Model loaded successfully: {model.name}")
