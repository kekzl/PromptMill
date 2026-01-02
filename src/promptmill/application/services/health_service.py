"""Health check service."""

from dataclasses import dataclass

from promptmill.application.use_cases.get_health_status import (
    GetHealthStatusUseCase,
    HealthStatus,
)


@dataclass
class HealthService:
    """High-level service for health checks.

    Provides a simple interface for health monitoring endpoints.
    """

    get_health_use_case: GetHealthStatusUseCase

    def get_status(self) -> HealthStatus:
        """Get current health status.

        Returns:
            HealthStatus dict with current application state.
        """
        return self.get_health_use_case.execute()

    def is_healthy(self) -> bool:
        """Quick health check.

        Returns:
            True if application is healthy.
        """
        try:
            status = self.get_status()
            return status["status"] == "healthy"
        except Exception:
            return False
