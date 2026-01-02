"""Application use cases."""

from promptmill.application.use_cases.delete_model import DeleteModelUseCase
from promptmill.application.use_cases.generate_prompt import GeneratePromptUseCase
from promptmill.application.use_cases.get_health_status import GetHealthStatusUseCase
from promptmill.application.use_cases.load_model import LoadModelUseCase
from promptmill.application.use_cases.select_model_by_vram import SelectModelByVRAMUseCase
from promptmill.application.use_cases.unload_model import UnloadModelUseCase

__all__ = [
    "DeleteModelUseCase",
    "GeneratePromptUseCase",
    "GetHealthStatusUseCase",
    "LoadModelUseCase",
    "SelectModelByVRAMUseCase",
    "UnloadModelUseCase",
]
