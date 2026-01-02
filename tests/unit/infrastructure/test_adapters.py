"""Tests for infrastructure adapters."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from promptmill.domain.entities.model import Model
from promptmill.domain.exceptions import ModelNotLoadedError
from promptmill.infrastructure.adapters.gpu_detector_adapter import NvidiaSmiAdapter
from promptmill.infrastructure.adapters.huggingface_adapter import HuggingFaceAdapter
from promptmill.infrastructure.adapters.llama_cpp_adapter import LlamaCppAdapter
from promptmill.infrastructure.adapters.role_repository_adapter import RoleRepositoryAdapter


class TestLlamaCppAdapter:
    """Tests for LlamaCppAdapter."""

    def test_initial_state(self) -> None:
        """Test adapter starts unloaded."""
        adapter = LlamaCppAdapter()
        assert adapter.is_loaded() is False
        assert adapter.get_loaded_model_path() is None

    def test_generate_not_loaded_raises(self) -> None:
        """Test generation raises when no model loaded."""
        adapter = LlamaCppAdapter()
        with pytest.raises(ModelNotLoadedError):
            list(adapter.generate("system", "user", 0.7, 256))

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent file raises."""
        adapter = LlamaCppAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load(
                str(tmp_path / "nonexistent.gguf"),
                n_gpu_layers=-1,
                context_length=4096,
            )

    def test_load_success(self, mock_model_file: Path, _patch_llama) -> None:
        """Test successful model loading."""
        adapter = LlamaCppAdapter()
        adapter.load(
            str(mock_model_file),
            n_gpu_layers=-1,
            context_length=4096,
        )

        assert adapter.is_loaded() is True
        assert adapter.get_loaded_model_path() == str(mock_model_file)

    def test_unload_success(self, mock_model_file: Path, _patch_llama) -> None:
        """Test successful model unloading."""
        adapter = LlamaCppAdapter()
        adapter.load(str(mock_model_file), n_gpu_layers=-1, context_length=4096)

        adapter.unload()

        assert adapter.is_loaded() is False
        assert adapter.get_loaded_model_path() is None

    def test_unload_when_not_loaded(self) -> None:
        """Test unloading when nothing loaded is safe."""
        adapter = LlamaCppAdapter()
        adapter.unload()  # Should not raise
        assert adapter.is_loaded() is False


class TestHuggingFaceAdapter:
    """Tests for HuggingFaceAdapter."""

    def test_get_model_path_not_found(
        self,
        tmp_path: Path,
        sample_model: Model,
    ) -> None:
        """Test get_model_path returns None when not downloaded."""
        adapter = HuggingFaceAdapter(tmp_path)
        path = adapter.get_model_path(sample_model)
        assert path is None

    def test_get_model_path_found(
        self,
        tmp_path: Path,
        sample_model: Model,
    ) -> None:
        """Test get_model_path returns path when downloaded."""
        model_file = tmp_path / sample_model.filename
        model_file.write_text("mock data")

        adapter = HuggingFaceAdapter(tmp_path)
        path = adapter.get_model_path(sample_model)

        assert path == model_file

    def test_delete_model_success(
        self,
        tmp_path: Path,
        sample_model: Model,
    ) -> None:
        """Test successful model deletion."""
        model_file = tmp_path / sample_model.filename
        model_file.write_text("mock data")

        adapter = HuggingFaceAdapter(tmp_path)
        result = adapter.delete_model(sample_model)

        assert result is True
        assert not model_file.exists()

    def test_delete_model_not_found(
        self,
        tmp_path: Path,
        sample_model: Model,
    ) -> None:
        """Test deleting non-existent model."""
        adapter = HuggingFaceAdapter(tmp_path)
        result = adapter.delete_model(sample_model)
        assert result is False

    def test_get_disk_usage(self, tmp_path: Path) -> None:
        """Test disk usage calculation."""
        # Create some files
        (tmp_path / "file1.gguf").write_bytes(b"x" * 1000)
        (tmp_path / "file2.gguf").write_bytes(b"x" * 2000)

        adapter = HuggingFaceAdapter(tmp_path)
        usage = adapter.get_disk_usage()

        assert usage == 3000

    def test_format_size(self, tmp_path: Path) -> None:
        """Test size formatting."""
        adapter = HuggingFaceAdapter(tmp_path)

        assert adapter.format_size(500) == "500.0 B"
        assert adapter.format_size(1024) == "1.0 KB"
        assert adapter.format_size(1024 * 1024) == "1.0 MB"
        assert adapter.format_size(1024 * 1024 * 1024) == "1.0 GB"


class TestNvidiaSmiAdapter:
    """Tests for NvidiaSmiAdapter."""

    def test_detect_no_nvidia_smi(self) -> None:
        """Test detection when nvidia-smi not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            adapter = NvidiaSmiAdapter()
            result = adapter.detect()

            assert result is None

    def test_detect_nvidia_smi_error(self) -> None:
        """Test detection when nvidia-smi returns error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            adapter = NvidiaSmiAdapter()
            result = adapter.detect()

            assert result is None

    def test_detect_success(self) -> None:
        """Test successful GPU detection."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GeForce RTX 4090, 24576, 535.154.05\n",
            )

            adapter = NvidiaSmiAdapter()
            result = adapter.detect()

            assert result is not None
            assert result.name == "NVIDIA GeForce RTX 4090"
            assert result.vram_mb == 24576
            assert result.driver_version == "535.154.05"

    def test_is_cuda_available_true(self) -> None:
        """Test CUDA availability check when GPU present."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="RTX 4090, 24576, 535.0\n",
            )

            adapter = NvidiaSmiAdapter()
            assert adapter.is_cuda_available() is True

    def test_is_cuda_available_false(self) -> None:
        """Test CUDA availability check when no GPU."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            adapter = NvidiaSmiAdapter()
            assert adapter.is_cuda_available() is False


class TestRoleRepositoryAdapter:
    """Tests for RoleRepositoryAdapter."""

    def test_get_all_roles(self) -> None:
        """Test getting all roles."""
        adapter = RoleRepositoryAdapter()
        roles = adapter.get_all()

        assert len(roles) > 0
        assert all(role.system_prompt for role in roles)

    def test_get_by_display_name(self) -> None:
        """Test getting role by display name."""
        adapter = RoleRepositoryAdapter()

        # Get first role
        all_roles = adapter.get_all()
        first_role = all_roles[0]

        # Look it up
        found = adapter.get_by_display_name(first_role.display_name)
        assert found is not None
        assert found.name == first_role.name

    def test_get_by_display_name_not_found(self) -> None:
        """Test getting non-existent role returns None."""
        adapter = RoleRepositoryAdapter()
        result = adapter.get_by_display_name("[Video] NonExistent")
        assert result is None

    def test_get_by_category(self) -> None:
        """Test getting roles by category."""
        from promptmill.domain.entities.role import RoleCategory

        adapter = RoleRepositoryAdapter()
        video_roles = adapter.get_by_category(RoleCategory.VIDEO)

        assert len(video_roles) > 0
        assert all(role.category == RoleCategory.VIDEO for role in video_roles)

    def test_get_display_names(self) -> None:
        """Test getting all display names."""
        adapter = RoleRepositoryAdapter()
        names = adapter.get_display_names()

        assert len(names) > 0
        assert all("[" in name and "]" in name for name in names)

    def test_count(self) -> None:
        """Test counting total roles."""
        adapter = RoleRepositoryAdapter()
        count = adapter.count()

        # Should have 102 roles based on documentation
        assert count == 102

    def test_get_categories(self) -> None:
        """Test getting available categories."""
        from promptmill.domain.entities.role import RoleCategory

        adapter = RoleRepositoryAdapter()
        categories = adapter.get_categories()

        assert RoleCategory.VIDEO in categories
        assert RoleCategory.IMAGE in categories
        assert RoleCategory.AUDIO in categories
        assert RoleCategory.THREE_D in categories
        assert RoleCategory.CREATIVE in categories
