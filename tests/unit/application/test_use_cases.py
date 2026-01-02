"""Tests for application use cases."""

from pathlib import Path
from threading import RLock
from unittest.mock import MagicMock

import pytest

from promptmill.application.use_cases.generate_prompt import GeneratePromptUseCase
from promptmill.application.use_cases.load_model import LoadModelUseCase
from promptmill.application.use_cases.select_model_by_vram import SelectModelByVRAMUseCase
from promptmill.application.use_cases.unload_model import UnloadModelUseCase
from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import Role
from promptmill.domain.exceptions import ModelNotLoadedError, RoleNotFoundError
from promptmill.domain.value_objects.prompt_request import PromptGenerationRequest
from promptmill.infrastructure.config.model_configs import MODEL_CONFIGS


class TestGeneratePromptUseCase:
    """Tests for GeneratePromptUseCase."""

    def test_generate_prompt_success(
        self,
        mock_llm: MagicMock,
        mock_role_repository: MagicMock,
        sample_role: Role,
    ) -> None:
        """Test successful prompt generation."""
        mock_llm.is_loaded.return_value = True

        use_case = GeneratePromptUseCase(
            llm=mock_llm,
            role_repository=mock_role_repository,
        )

        request = PromptGenerationRequest(
            user_input="A sunset scene",
            role_display_name=sample_role.display_name,
            temperature=0.7,
            max_tokens=256,
        )

        result = list(use_case.execute(request))
        assert result == ["Test ", "response ", "content"]
        mock_llm.generate.assert_called_once()

    def test_generate_prompt_model_not_loaded(
        self,
        mock_llm: MagicMock,
        mock_role_repository: MagicMock,
        sample_role: Role,
    ) -> None:
        """Test generation fails when model not loaded."""
        mock_llm.is_loaded.return_value = False

        use_case = GeneratePromptUseCase(
            llm=mock_llm,
            role_repository=mock_role_repository,
        )

        request = PromptGenerationRequest(
            user_input="A sunset scene",
            role_display_name=sample_role.display_name,
            temperature=0.7,
            max_tokens=256,
        )

        with pytest.raises(ModelNotLoadedError):
            list(use_case.execute(request))

    def test_generate_prompt_role_not_found(
        self,
        mock_llm: MagicMock,
        mock_role_repository: MagicMock,
    ) -> None:
        """Test generation fails when role not found."""
        mock_llm.is_loaded.return_value = True
        mock_role_repository.get_by_display_name.return_value = None

        use_case = GeneratePromptUseCase(
            llm=mock_llm,
            role_repository=mock_role_repository,
        )

        request = PromptGenerationRequest(
            user_input="A sunset scene",
            role_display_name="[Video] NonExistent",
            temperature=0.7,
            max_tokens=256,
        )

        with pytest.raises(RoleNotFoundError):
            list(use_case.execute(request))


class TestLoadModelUseCase:
    """Tests for LoadModelUseCase."""

    def test_load_model_success(
        self,
        mock_llm: MagicMock,
        mock_model_repository: MagicMock,
        sample_model: Model,
        tmp_path: Path,
        model_lock: RLock,
    ) -> None:
        """Test successful model loading."""
        use_case = LoadModelUseCase(
            llm=mock_llm,
            model_repository=mock_model_repository,
            lock=model_lock,
        )

        use_case.execute(sample_model, tmp_path)

        mock_model_repository.download_model.assert_called_once_with(sample_model)
        mock_llm.load.assert_called_once()

    def test_load_model_already_loaded(
        self,
        mock_llm: MagicMock,
        mock_model_repository: MagicMock,
        sample_model: Model,
        tmp_path: Path,
        model_lock: RLock,
    ) -> None:
        """Test loading already loaded model is a no-op."""
        model_path = str(tmp_path / sample_model.filename)
        mock_llm.get_loaded_model_path.return_value = model_path

        use_case = LoadModelUseCase(
            llm=mock_llm,
            model_repository=mock_model_repository,
            lock=model_lock,
        )

        use_case.execute(sample_model, tmp_path)

        mock_llm.load.assert_not_called()

    def test_load_model_unloads_previous(
        self,
        mock_llm: MagicMock,
        mock_model_repository: MagicMock,
        sample_model: Model,
        tmp_path: Path,
        model_lock: RLock,
    ) -> None:
        """Test loading new model unloads previous."""
        mock_llm.is_loaded.return_value = True
        mock_llm.get_loaded_model_path.return_value = "/other/model.gguf"

        use_case = LoadModelUseCase(
            llm=mock_llm,
            model_repository=mock_model_repository,
            lock=model_lock,
        )

        use_case.execute(sample_model, tmp_path)

        mock_llm.unload.assert_called_once()
        mock_llm.load.assert_called_once()


class TestUnloadModelUseCase:
    """Tests for UnloadModelUseCase."""

    def test_unload_model_success(
        self,
        mock_llm: MagicMock,
        model_lock: RLock,
    ) -> None:
        """Test successful model unloading."""
        mock_llm.is_loaded.return_value = True

        use_case = UnloadModelUseCase(llm=mock_llm, lock=model_lock)
        result = use_case.execute()

        assert result is True
        mock_llm.unload.assert_called_once()

    def test_unload_model_not_loaded(
        self,
        mock_llm: MagicMock,
        model_lock: RLock,
    ) -> None:
        """Test unloading when no model loaded."""
        mock_llm.is_loaded.return_value = False

        use_case = UnloadModelUseCase(llm=mock_llm, lock=model_lock)
        result = use_case.execute()

        assert result is False
        mock_llm.unload.assert_not_called()


class TestSelectModelByVRAMUseCase:
    """Tests for SelectModelByVRAMUseCase."""

    def test_select_cpu_model_no_gpu(
        self,
        mock_gpu_detector_no_gpu: MagicMock,
    ) -> None:
        """Test CPU model selection when no GPU."""
        use_case = SelectModelByVRAMUseCase(gpu_detector=mock_gpu_detector_no_gpu)
        model, gpu_info = use_case.execute()

        assert model.key == "cpu_only"
        assert gpu_info is None

    def test_select_24gb_model(
        self,
        mock_gpu_detector: MagicMock,
    ) -> None:
        """Test 24GB model selection for high VRAM GPU."""
        mock_gpu_detector.detect.return_value = GPUInfo(
            name="RTX 4090",
            vram_mb=24576,  # 24GB
            driver_version="535.0",
        )

        use_case = SelectModelByVRAMUseCase(gpu_detector=mock_gpu_detector)
        model, gpu_info = use_case.execute()

        assert model.key == "24gb_vram"
        assert gpu_info is not None
        assert gpu_info.vram_gb == 24.0

    def test_select_8gb_model(
        self,
        mock_gpu_detector: MagicMock,
    ) -> None:
        """Test 8GB model selection for mid-range GPU."""
        mock_gpu_detector.detect.return_value = GPUInfo(
            name="RTX 4060",
            vram_mb=8192,  # 8GB
            driver_version="535.0",
        )

        use_case = SelectModelByVRAMUseCase(gpu_detector=mock_gpu_detector)
        model, gpu_info = use_case.execute()

        assert model.key == "8gb_vram"

    def test_select_4gb_model(
        self,
        mock_gpu_detector: MagicMock,
    ) -> None:
        """Test 4GB model selection for low VRAM GPU."""
        mock_gpu_detector.detect.return_value = GPUInfo(
            name="GTX 1650",
            vram_mb=4096,  # 4GB
            driver_version="535.0",
        )

        use_case = SelectModelByVRAMUseCase(gpu_detector=mock_gpu_detector)
        model, gpu_info = use_case.execute()

        assert model.key == "4gb_vram"
