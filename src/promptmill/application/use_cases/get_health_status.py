"""Get health status use case."""

from dataclasses import dataclass
from typing import TypedDict

from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort


class HealthStatus(TypedDict):
    """Health status response structure."""

    status: str
    version: str
    model_loaded: bool
    model_path: str | None
    roles_count: int
    disk_usage_bytes: int


@dataclass(slots=True)
class GetHealthStatusUseCase:
    """Use case for getting application health status.

    This use case gathers health information from various components
    for monitoring and debugging purposes.
    """

    llm: LLMPort
    model_repository: ModelRepositoryPort
    role_repository: RoleRepositoryPort
    version: str

    def execute(self) -> HealthStatus:
        """Execute the health check use case.

        Returns:
            HealthStatus dict with current application state.
        """
        return HealthStatus(
            status="healthy",
            version=self.version,
            model_loaded=self.llm.is_loaded(),
            model_path=self.llm.get_loaded_model_path(),
            roles_count=self.role_repository.count(),
            disk_usage_bytes=self.model_repository.get_disk_usage(),
        )
