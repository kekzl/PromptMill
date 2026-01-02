"""NVIDIA SMI GPU detector adapter implementing GPUDetectorPort."""

import logging
import subprocess
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing import TypeVar

    def override(func: TypeVar("F")) -> TypeVar("F"):  # type: ignore[misc]
        return func  # type: ignore[return-value]


from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort

logger = logging.getLogger(__name__)

# Timeout for nvidia-smi command
GPU_DETECTION_TIMEOUT = 5


class NvidiaSmiAdapter(GPUDetectorPort):
    """Adapter for NVIDIA GPU detection via nvidia-smi.

    This adapter implements the GPUDetectorPort interface using
    the nvidia-smi command-line tool to detect GPU information.
    """

    __slots__ = ("_timeout",)

    def __init__(self, timeout: int = GPU_DETECTION_TIMEOUT) -> None:
        """Initialize the adapter.

        Args:
            timeout: Timeout in seconds for nvidia-smi command.
        """
        self._timeout = timeout

    @override
    def detect(self) -> GPUInfo | None:
        """Detect GPU and return information.

        Returns:
            GPUInfo if a GPU is detected, None if no GPU available.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )

            if result.returncode != 0 or not result.stdout.strip():
                logger.info("No GPU detected via nvidia-smi")
                return None

            # Parse first GPU (in case of multi-GPU system)
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 3:
                logger.warning(f"Unexpected nvidia-smi output format: {line}")
                return None

            name = parts[0]
            vram_mb = int(parts[1])
            driver_version = parts[2]

            gpu_info = GPUInfo(
                name=name,
                vram_mb=vram_mb,
                driver_version=driver_version,
            )

            logger.info(f"GPU detected: {gpu_info}")
            return gpu_info

        except FileNotFoundError:
            logger.debug("nvidia-smi not found - no NVIDIA GPU available")
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"GPU detection timed out after {self._timeout} seconds")
            return None

        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing GPU info: {e}")
            return None

        except OSError as e:
            logger.debug(f"OS error during GPU detection: {e}")
            return None

    @override
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA runtime is available.
        """
        gpu_info = self.detect()
        return gpu_info is not None and gpu_info.is_available
