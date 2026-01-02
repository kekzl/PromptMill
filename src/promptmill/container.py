"""Dependency Injection Container for PromptMill.

This module provides manual dependency injection using pure Python.
All dependencies are lazily initialized and properly wired together.
"""

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from promptmill.application.services.health_service import HealthService
from promptmill.application.services.model_service import ModelService
from promptmill.application.services.prompt_service import PromptService
from promptmill.application.use_cases.delete_model import DeleteModelUseCase
from promptmill.application.use_cases.generate_prompt import GeneratePromptUseCase
from promptmill.application.use_cases.get_health_status import GetHealthStatusUseCase
from promptmill.application.use_cases.load_model import LoadModelUseCase
from promptmill.application.use_cases.select_model_by_vram import SelectModelByVRAMUseCase
from promptmill.application.use_cases.unload_model import UnloadModelUseCase
from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort
from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort
from promptmill.infrastructure.adapters.gpu_detector_adapter import NvidiaSmiAdapter
from promptmill.infrastructure.adapters.huggingface_adapter import HuggingFaceAdapter
from promptmill.infrastructure.adapters.llama_cpp_adapter import LlamaCppAdapter
from promptmill.infrastructure.adapters.role_repository_adapter import RoleRepositoryAdapter
from promptmill.infrastructure.config.settings import Settings
from promptmill.presentation.gradio_app import GradioApp


@dataclass
class Container:
    """Manual dependency injection container.

    This container manages the creation and wiring of all application
    components. Dependencies are lazily initialized on first access.

    Usage:
        container = Container(settings=Settings())
        app = container.gradio_app
        app.launch(host, port)
    """

    settings: Settings

    # Shared threading lock
    _lock: RLock = field(default_factory=RLock, init=False)

    # Cached instances (lazy initialization)
    _llm_adapter: LLMPort | None = field(default=None, init=False)
    _model_repository: ModelRepositoryPort | None = field(default=None, init=False)
    _gpu_detector: GPUDetectorPort | None = field(default=None, init=False)
    _role_repository: RoleRepositoryPort | None = field(default=None, init=False)
    _detected_gpu: GPUInfo | None = field(default=None, init=False)
    _selected_model: Model | None = field(default=None, init=False)
    _gpu_detection_done: bool = field(default=False, init=False)

    # =========================================================================
    # Infrastructure Adapters (Ports implementations)
    # =========================================================================

    @property
    def llm(self) -> LLMPort:
        """Get LLM adapter instance."""
        if self._llm_adapter is None:
            self._llm_adapter = LlamaCppAdapter(
                chat_format=self.settings.default_chat_format,
                batch_size=self.settings.default_batch_size,
            )
        return self._llm_adapter

    @property
    def model_repository(self) -> ModelRepositoryPort:
        """Get model repository adapter instance."""
        if self._model_repository is None:
            self.settings.ensure_models_dir()
            self._model_repository = HuggingFaceAdapter(self.settings.models_dir)
        return self._model_repository

    @property
    def gpu_detector(self) -> GPUDetectorPort:
        """Get GPU detector adapter instance."""
        if self._gpu_detector is None:
            self._gpu_detector = NvidiaSmiAdapter(
                timeout=self.settings.gpu_detection_timeout
            )
        return self._gpu_detector

    @property
    def role_repository(self) -> RoleRepositoryPort:
        """Get role repository adapter instance."""
        if self._role_repository is None:
            self._role_repository = RoleRepositoryAdapter()
        return self._role_repository

    # =========================================================================
    # GPU Detection and Model Selection
    # =========================================================================

    def _ensure_gpu_detection(self) -> None:
        """Ensure GPU detection has been performed."""
        if not self._gpu_detection_done:
            select_use_case = SelectModelByVRAMUseCase(gpu_detector=self.gpu_detector)
            self._selected_model, self._detected_gpu = select_use_case.execute()
            self._gpu_detection_done = True

    @property
    def detected_gpu(self) -> GPUInfo | None:
        """Get detected GPU information."""
        self._ensure_gpu_detection()
        return self._detected_gpu

    @property
    def default_model(self) -> Model:
        """Get the default model based on detected GPU."""
        self._ensure_gpu_detection()
        return self._selected_model

    # =========================================================================
    # Use Cases
    # =========================================================================

    @property
    def generate_prompt_use_case(self) -> GeneratePromptUseCase:
        """Create GeneratePromptUseCase instance."""
        return GeneratePromptUseCase(
            llm=self.llm,
            role_repository=self.role_repository,
        )

    @property
    def load_model_use_case(self) -> LoadModelUseCase:
        """Create LoadModelUseCase instance."""
        return LoadModelUseCase(
            llm=self.llm,
            model_repository=self.model_repository,
            lock=self._lock,
        )

    @property
    def unload_model_use_case(self) -> UnloadModelUseCase:
        """Create UnloadModelUseCase instance."""
        return UnloadModelUseCase(
            llm=self.llm,
            lock=self._lock,
        )

    @property
    def select_model_use_case(self) -> SelectModelByVRAMUseCase:
        """Create SelectModelByVRAMUseCase instance."""
        return SelectModelByVRAMUseCase(gpu_detector=self.gpu_detector)

    @property
    def delete_model_use_case(self) -> DeleteModelUseCase:
        """Create DeleteModelUseCase instance."""
        return DeleteModelUseCase(
            llm=self.llm,
            model_repository=self.model_repository,
            lock=self._lock,
        )

    @property
    def get_health_use_case(self) -> GetHealthStatusUseCase:
        """Create GetHealthStatusUseCase instance."""
        from promptmill import __version__

        return GetHealthStatusUseCase(
            llm=self.llm,
            model_repository=self.model_repository,
            role_repository=self.role_repository,
            version=__version__,
        )

    # =========================================================================
    # Application Services
    # =========================================================================

    @property
    def prompt_service(self) -> PromptService:
        """Create PromptService instance."""
        return PromptService(
            generate_prompt_use_case=self.generate_prompt_use_case,
            load_model_use_case=self.load_model_use_case,
            unload_model_use_case=self.unload_model_use_case,
            models_dir=self.settings.models_dir,
            unload_delay_seconds=self.settings.unload_delay_seconds,
        )

    @property
    def model_service(self) -> ModelService:
        """Create ModelService instance."""
        return ModelService(
            load_model_use_case=self.load_model_use_case,
            unload_model_use_case=self.unload_model_use_case,
            delete_model_use_case=self.delete_model_use_case,
            select_model_use_case=self.select_model_use_case,
            model_repository=self.model_repository,
            models_dir=self.settings.models_dir,
        )

    @property
    def health_service(self) -> HealthService:
        """Create HealthService instance."""
        return HealthService(get_health_use_case=self.get_health_use_case)

    # =========================================================================
    # Presentation Layer
    # =========================================================================

    @property
    def gradio_app(self) -> GradioApp:
        """Create GradioApp instance."""
        # Determine assets directory relative to project
        assets_dir = Path(__file__).parent.parent.parent / "assets"
        if not assets_dir.exists():
            # Try relative to working directory
            assets_dir = Path("assets")

        return GradioApp(
            prompt_service=self.prompt_service,
            model_service=self.model_service,
            health_service=self.health_service,
            assets_dir=assets_dir,
            gpu_info=self.detected_gpu,
            default_model=self.default_model,
        )

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def shutdown(self) -> None:
        """Clean shutdown of all components."""
        if self._llm_adapter is not None:
            self._llm_adapter.unload()
