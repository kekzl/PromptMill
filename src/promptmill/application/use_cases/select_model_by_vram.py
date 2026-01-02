"""Select model by VRAM use case."""

import logging
from dataclasses import dataclass

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort
from promptmill.infrastructure.config.model_configs import MODEL_CONFIGS

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

        # Select model based on VRAM using pattern matching
        model = self._select_by_vram(vram_gb)
        logger.info(f"Selected model: {model.name}")

        return model, gpu_info

    def _select_by_vram(self, vram_gb: float) -> Model:
        """Select model based on available VRAM.

        Args:
            vram_gb: Available VRAM in gigabytes.

        Returns:
            Optimal model for the available VRAM.
        """
        match vram_gb:
            case v if v >= 20:
                return MODEL_CONFIGS["24gb_vram"]
            case v if v >= 14:
                return MODEL_CONFIGS["16gb_vram"]
            case v if v >= 10:
                return MODEL_CONFIGS["12gb_vram"]
            case v if v >= 7:
                return MODEL_CONFIGS["8gb_vram"]
            case v if v >= 5:
                return MODEL_CONFIGS["6gb_vram"]
            case v if v >= 3:
                return MODEL_CONFIGS["4gb_vram"]
            case _:
                return MODEL_CONFIGS["cpu_only"]
