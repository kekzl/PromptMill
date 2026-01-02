"""Delete model use case."""

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from promptmill.domain.entities.model import Model
from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeleteModelUseCase:
    """Use case for deleting a downloaded model.

    This use case:
    1. Checks if the model to delete is currently loaded
    2. Unloads it if necessary
    3. Deletes the model file
    """

    llm: LLMPort
    model_repository: ModelRepositoryPort
    lock: RLock

    def execute(self, model: Model, models_dir: Path) -> bool:
        """Execute the model deletion use case.

        Args:
            model: The model to delete.
            models_dir: Directory where models are stored.

        Returns:
            True if model was deleted, False if it wasn't found.
        """
        with self.lock:
            # Check if this model is currently loaded
            current_path = self.llm.get_loaded_model_path()
            model_path = str(models_dir / model.filename)

            if current_path == model_path:
                logger.info(f"Model {model.name} is loaded, unloading first")
                self.llm.unload()

            # Delete the model file
            deleted = self.model_repository.delete_model(model)

            if deleted:
                logger.info(f"Model deleted: {model.name}")
            else:
                logger.info(f"Model not found for deletion: {model.name}")

            return deleted
