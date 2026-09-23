"""Select model by VRAM use case."""

import logging
from dataclasses import dataclass

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort
from promptmill.infrastructure.config.model_configs import MODEL_CONFIGS, select_model_by_vram

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelectModelByVRAMUseCase:
    """Use case for selecting the optimal model based on available VRAM.

    This use case:
    1. Detects available GPU and VRAM
    2. Selects the best model configuration for the hardware
    """

    gpu_detector: GPUDetectorPort

    def execute(self) -> tuple[Model, GPUInfo | None]:
        """Execute the model selection use case.

        Returns:
            Tuple of (selected_model, gpu_info).
            gpu_info is None if no GPU was detected.
        """
        # Detect GPU
        gpu_info = self.gpu_detector.detect()

        if gpu_info is None or not gpu_info.is_available:
            logger.info("No GPU detected, selecting CPU-only model")
            return MODEL_CONFIGS["cpu_only"], gpu_info

        vram_gb = gpu_info.vram_gb
        logger.info(f"GPU detected: {gpu_info.name} with {vram_gb:.1f} GB VRAM")

        model = select_model_by_vram(gpu_info.vram_mb)
        logger.info(f"Selected model: {model.name}")

        return model, gpu_info
