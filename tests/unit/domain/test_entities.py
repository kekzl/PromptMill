"""Tests for domain entities."""

from pathlib import Path

import pytest

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import Role, RoleCategory


class TestModel:
    """Tests for Model entity."""

    def test_model_creation(self, sample_model: Model) -> None:
        """Test Model entity creation."""
        assert sample_model.key == "test_model"
        assert sample_model.name == "Test Model (8GB VRAM)"
        assert sample_model.context_length == 4096
        assert sample_model.n_gpu_layers == -1

    def test_model_is_frozen(self, sample_model: Model) -> None:
        """Test that Model is immutable."""
        with pytest.raises(AttributeError):
            sample_model.name = "New Name"  # type: ignore

    def test_get_local_path(self, sample_model: Model, tmp_path: Path) -> None:
        """Test getting local file path."""
        path = sample_model.get_local_path(tmp_path)
        assert path == tmp_path / "test-model.gguf"

    def test_is_downloaded_false(self, sample_model: Model, tmp_path: Path) -> None:
        """Test is_downloaded returns False when file doesn't exist."""
        assert sample_model.is_downloaded(tmp_path) is False

    def test_is_downloaded_true(self, sample_model: Model, tmp_path: Path) -> None:
        """Test is_downloaded returns True when file exists."""
        model_file = tmp_path / sample_model.filename
        model_file.write_text("mock data")
        assert sample_model.is_downloaded(tmp_path) is True


class TestRole:
    """Tests for Role entity."""

    def test_role_creation(self, sample_role: Role) -> None:
        """Test Role entity creation."""
        assert sample_role.name == "TestRole"
        assert sample_role.category == RoleCategory.VIDEO
        assert "test assistant" in sample_role.system_prompt

    def test_role_is_frozen(self, sample_role: Role) -> None:
        """Test that Role is immutable."""
        with pytest.raises(AttributeError):
            sample_role.name = "New Name"  # type: ignore

    def test_display_name(self, sample_role: Role) -> None:
        """Test display name formatting."""
        assert sample_role.display_name == "[Video] TestRole"

    def test_parse_display_name_valid(self) -> None:
        """Test parsing valid display name."""
        category, name = Role.parse_display_name("[Video] Midjourney")
        assert category == "Video"
        assert name == "Midjourney"

    def test_parse_display_name_invalid(self) -> None:
        """Test parsing invalid display name."""
        with pytest.raises(ValueError):
            Role.parse_display_name("Invalid Format")


class TestRoleCategory:
    """Tests for RoleCategory enum."""

    def test_all_categories(self) -> None:
        """Test all category values."""
        assert RoleCategory.VIDEO.value == "Video"
        assert RoleCategory.IMAGE.value == "Image"
        assert RoleCategory.AUDIO.value == "Audio"
        assert RoleCategory.THREE_D.value == "3D"
        assert RoleCategory.CREATIVE.value == "Creative"

    def test_from_string_valid(self) -> None:
        """Test from_string with valid input."""
        assert RoleCategory.from_string("Video") == RoleCategory.VIDEO
        assert RoleCategory.from_string("video") == RoleCategory.VIDEO  # Case insensitive
        assert RoleCategory.from_string("3D") == RoleCategory.THREE_D

    def test_from_string_invalid(self) -> None:
        """Test from_string with invalid input defaults to CREATIVE."""
        assert RoleCategory.from_string("Unknown") == RoleCategory.CREATIVE


class TestGPUInfo:
    """Tests for GPUInfo value object."""

    def test_gpu_info_creation(self, sample_gpu_info: GPUInfo) -> None:
        """Test GPUInfo creation."""
        assert sample_gpu_info.name == "NVIDIA GeForce RTX 4090"
        assert sample_gpu_info.vram_mb == 24576
        assert sample_gpu_info.driver_version == "535.154.05"

    def test_gpu_info_is_frozen(self, sample_gpu_info: GPUInfo) -> None:
        """Test that GPUInfo is immutable."""
        with pytest.raises(AttributeError):
            sample_gpu_info.vram_mb = 0  # type: ignore

    def test_vram_gb(self, sample_gpu_info: GPUInfo) -> None:
        """Test VRAM conversion to GB."""
        assert sample_gpu_info.vram_gb == 24.0

    def test_is_available_true(self, sample_gpu_info: GPUInfo) -> None:
        """Test is_available for GPU with VRAM."""
        assert sample_gpu_info.is_available is True

    def test_is_available_false(self, cpu_only_gpu_info: GPUInfo) -> None:
        """Test is_available for CPU-only."""
        assert cpu_only_gpu_info.is_available is False

    def test_cpu_only_factory(self) -> None:
        """Test cpu_only factory method."""
        gpu = GPUInfo.cpu_only()
        assert gpu.name == "CPU"
        assert gpu.vram_mb == 0
        assert gpu.is_available is False

    def test_str_with_gpu(self, sample_gpu_info: GPUInfo) -> None:
        """Test string representation with GPU."""
        s = str(sample_gpu_info)
        assert "RTX 4090" in s
        assert "24.0 GB" in s

    def test_str_cpu_only(self, cpu_only_gpu_info: GPUInfo) -> None:
        """Test string representation for CPU-only."""
        s = str(cpu_only_gpu_info)
        assert "CPU Only" in s
