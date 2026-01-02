"""GPU detector port (interface)."""

from abc import ABC, abstractmethod

from promptmill.domain.entities.gpu_info import GPUInfo


class GPUDetectorPort(ABC):
    """Port for GPU detection operations.

    This defines the interface for detecting available GPU hardware
    and retrieving its specifications.
    """

    @abstractmethod
    def detect(self) -> GPUInfo | None:
        """Detect GPU and return information.

        Returns:
            GPUInfo if a GPU is detected, None if no GPU available.
            Returns None rather than GPUInfo.cpu_only() to distinguish
            between "no GPU" and "detection failed".
        """
        ...

    @abstractmethod
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA runtime is available.
        """
        ...
