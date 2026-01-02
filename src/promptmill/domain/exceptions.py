"""Domain exceptions."""


class PromptMillError(Exception):
    """Base exception for all PromptMill errors."""


class DomainError(PromptMillError):
    """Base exception for domain layer errors."""


class ValidationError(DomainError):
    """Raised when input validation fails."""


class RoleNotFoundError(DomainError):
    """Raised when a requested role doesn't exist."""

    def __init__(self, role_name: str) -> None:
        self.role_name = role_name
        super().__init__(f"Role not found: {role_name}")


class ModelNotFoundError(DomainError):
    """Raised when a requested model doesn't exist."""

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        super().__init__(f"Model not found: {model_key}")


class ModelNotLoadedError(DomainError):
    """Raised when trying to generate without a loaded model."""

    def __init__(self) -> None:
        super().__init__("No model is currently loaded")


class InfrastructureError(PromptMillError):
    """Base exception for infrastructure layer errors."""


class ModelDownloadError(InfrastructureError):
    """Raised when model download fails."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to download model '{model_name}': {reason}")


class ModelLoadError(InfrastructureError):
    """Raised when model loading fails."""

    def __init__(self, model_path: str, reason: str) -> None:
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Failed to load model from '{model_path}': {reason}")


class GPUDetectionError(InfrastructureError):
    """Raised when GPU detection fails unexpectedly."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"GPU detection failed: {reason}")


class InsufficientSpaceError(InfrastructureError):
    """Raised when there's not enough disk space for an operation."""

    def __init__(self, required_bytes: int, available_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        required_gb = required_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        super().__init__(
            f"Insufficient disk space: {required_gb:.1f} GB required, "
            f"{available_gb:.1f} GB available"
        )
