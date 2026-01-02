"""Domain layer - Pure business logic with no external dependencies."""

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import Role, RoleCategory
from promptmill.domain.value_objects.prompt_request import PromptGenerationRequest
from promptmill.domain.value_objects.prompt_result import PromptGenerationResult

__all__ = [
    "GPUInfo",
    "Model",
    "PromptGenerationRequest",
    "PromptGenerationResult",
    "Role",
    "RoleCategory",
]
