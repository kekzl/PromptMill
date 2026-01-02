"""Unload model use case."""

import logging
from dataclasses import dataclass
from threading import RLock

from promptmill.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UnloadModelUseCase:
    """Use case for unloading the current model from memory.

    This use case handles thread-safe model unloading to free
    GPU/CPU memory.
    """

    llm: LLMPort
    lock: RLock

    def execute(self) -> bool:
        """Execute the model unloading use case.

        Returns:
            True if a model was unloaded, False if no model was loaded.
        """
        with self.lock:
            if not self.llm.is_loaded():
                logger.debug("No model loaded to unload")
                return False

            logger.info("Unloading model")
            self.llm.unload()
            logger.info("Model unloaded successfully")
            return True
