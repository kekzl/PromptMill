"""Domain ports (interfaces) for dependency inversion."""

from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort
from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort

__all__ = [
    "GPUDetectorPort",
    "LLMPort",
    "ModelRepositoryPort",
    "RoleRepositoryPort",
]
