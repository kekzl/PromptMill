"""GPU information value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Value object representing GPU information.

    Attributes:
        name: GPU model name (e.g., "NVIDIA GeForce RTX 4090").
        vram_mb: Total VRAM in megabytes.
        driver_version: NVIDIA driver version string.
    """

    name: str
    vram_mb: int
    driver_version: str

    @property
    def vram_gb(self) -> float:
        """Get VRAM in gigabytes.

        Returns:
            VRAM converted to GB.
        """
        return self.vram_mb / 1024

    @property
    def is_available(self) -> bool:
        """Check if this represents an available GPU.

        Returns:
            True if GPU has VRAM (i.e., is not CPU-only).
        """
        return self.vram_mb > 0

    @classmethod
    def cpu_only(cls) -> "GPUInfo":
        """Create a GPUInfo representing CPU-only mode.

        Returns:
            GPUInfo with zero VRAM indicating no GPU.
        """
        return cls(name="CPU", vram_mb=0, driver_version="N/A")

    def __str__(self) -> str:
        """Human-readable representation."""
        if not self.is_available:
            return "CPU Only (No GPU)"
        return f"{self.name} ({self.vram_gb:.1f} GB VRAM)"
