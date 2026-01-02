"""Domain entities."""

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import Role, RoleCategory

__all__ = ["GPUInfo", "Model", "Role", "RoleCategory"]
