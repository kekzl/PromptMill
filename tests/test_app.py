"""Unit tests for PromptMill application."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the app module (we import specific functions to avoid loading the full Gradio UI)
import app


class TestConfiguration:
    """Test configuration constants."""

    def test_version_format(self):
        """Test version string format."""
        assert isinstance(app.__version__, str)
        parts = app.__version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_model_configs_exist(self):
        """Test that model configurations are defined."""
        assert len(app.MODEL_CONFIGS) > 0
        assert "CPU Only (2-4GB RAM)" in app.MODEL_CONFIGS

    def test_model_config_structure(self):
        """Test that each model config has required fields."""
        required_fields = ["repo", "file", "description", "vram", "n_ctx"]
        for model_key, config in app.MODEL_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Missing {field} in {model_key}"

    def test_configuration_constants(self):
        """Test configuration constants are set."""
        assert app.MAX_PROMPT_LENGTH > 0
        assert app.MIN_TEMPERATURE < app.MAX_TEMPERATURE
        assert app.MIN_TOKENS < app.MAX_TOKENS
        assert app.DEFAULT_BATCH_SIZE > 0
        assert app.UNLOAD_DELAY_SECONDS > 0


class TestGPUDetection:
    """Test GPU detection functionality."""

    @patch("subprocess.run")
    def test_detect_gpu_with_nvidia(self, mock_run):
        """Test GPU detection with NVIDIA GPU present."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="8192, NVIDIA GeForce RTX 3070\n"
        )
        has_gpu, vram_mb, gpu_name = app.detect_gpu()
        assert has_gpu is True
        assert vram_mb == 8192
        assert "RTX 3070" in gpu_name

    @patch("subprocess.run")
    def test_detect_gpu_no_nvidia(self, mock_run):
        """Test GPU detection without NVIDIA GPU."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        has_gpu, vram_mb, gpu_name = app.detect_gpu()
        assert has_gpu is False
        assert vram_mb == 0
        assert gpu_name == ""

    @patch("subprocess.run")
    def test_detect_gpu_nvidia_smi_not_found(self, mock_run):
        """Test GPU detection when nvidia-smi is not found."""
        mock_run.side_effect = FileNotFoundError()
        has_gpu, vram_mb, gpu_name = app.detect_gpu()
        assert has_gpu is False
        assert vram_mb == 0
        assert gpu_name == ""

    @patch("subprocess.run")
    def test_detect_gpu_timeout(self, mock_run):
        """Test GPU detection timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 5)
        has_gpu, vram_mb, gpu_name = app.detect_gpu()
        assert has_gpu is False
        assert vram_mb == 0


class TestModelSelection:
    """Test model selection by VRAM."""

    def test_select_model_cpu(self):
        """Test model selection for CPU (low VRAM)."""
        result = app.select_model_by_vram(0)
        assert "CPU Only" in result

    def test_select_model_4gb(self):
        """Test model selection for 4GB VRAM."""
        result = app.select_model_by_vram(4096)
        assert "4GB" in result

    def test_select_model_8gb(self):
        """Test model selection for 8GB VRAM."""
        result = app.select_model_by_vram(8192)
        assert "8GB" in result

    def test_select_model_24gb(self):
        """Test model selection for 24GB+ VRAM."""
        result = app.select_model_by_vram(24576)
        assert "24GB" in result


class TestRoleSystem:
    """Test role definitions and parsing."""

    def test_roles_defined(self):
        """Test that roles are defined."""
        assert len(app.ROLES) > 0

    def test_role_structure(self):
        """Test that each role has required fields."""
        required_fields = ["category", "name", "description", "system_prompt"]
        for role_id, role_data in app.ROLES.items():
            for field in required_fields:
                assert field in role_data, f"Missing {field} in role {role_id}"

    def test_role_categories(self):
        """Test that roles have valid categories."""
        valid_categories = {"Video", "Image", "Audio", "3D", "Creative"}
        for role_id, role_data in app.ROLES.items():
            assert role_data["category"] in valid_categories, f"Invalid category for {role_id}"

    def test_get_role_choices(self):
        """Test getting role choices for dropdown."""
        choices = app.get_role_choices()
        assert len(choices) > 0
        assert all("[" in choice and "]" in choice for choice in choices)

    def test_parse_role_choice(self):
        """Test parsing role choice from dropdown format."""
        result = app.parse_role_choice("[Video] Wan2.1")
        assert result == "Wan2.1"

    def test_parse_role_choice_without_bracket(self):
        """Test parsing role choice without bracket format."""
        result = app.parse_role_choice("Wan2.1")
        assert result == "Wan2.1"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_size_bytes(self):
        """Test formatting bytes."""
        assert app.format_size(500) == "500.0 B"

    def test_format_size_kilobytes(self):
        """Test formatting kilobytes."""
        result = app.format_size(1536)
        assert "KB" in result

    def test_format_size_megabytes(self):
        """Test formatting megabytes."""
        result = app.format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_size_gigabytes(self):
        """Test formatting gigabytes."""
        result = app.format_size(5 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_check_model_exists_nonexistent(self):
        """Test checking for nonexistent model."""
        with patch.object(app.MODELS_DIR, "__truediv__") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = app.check_model_exists("CPU Only (2-4GB RAM)")
            assert result is False


class TestInputValidation:
    """Test input validation in generate_prompt."""

    def test_empty_input_validation(self):
        """Test that empty input is rejected."""
        gen = app.generate_prompt("", "[Video] Wan2.1", "CPU Only (2-4GB RAM)", 0.7, 256, 0)
        result = next(gen)
        assert "Please enter" in result

    def test_whitespace_input_validation(self):
        """Test that whitespace-only input is rejected."""
        gen = app.generate_prompt("   ", "[Video] Wan2.1", "CPU Only (2-4GB RAM)", 0.7, 256, 0)
        result = next(gen)
        assert "Please enter" in result

    def test_input_too_long(self):
        """Test that overly long input is rejected."""
        long_input = "x" * (app.MAX_PROMPT_LENGTH + 1)
        gen = app.generate_prompt(
            long_input, "[Video] Wan2.1", "CPU Only (2-4GB RAM)", 0.7, 256, 0
        )
        result = next(gen)
        assert "too long" in result.lower()

    def test_unknown_role_validation(self):
        """Test that unknown role is rejected."""
        gen = app.generate_prompt(
            "Test idea", "[Video] NonexistentRole", "CPU Only (2-4GB RAM)", 0.7, 256, 0
        )
        result = next(gen)
        assert "Unknown role" in result


class TestModelManagement:
    """Test model management functions."""

    def test_unload_model_when_none(self):
        """Test unloading when no model is loaded."""
        app.llm = None
        app.current_model_key = None
        # Should not raise an exception
        app.unload_model()
        assert app.llm is None

    def test_cancel_unload_timer_when_none(self):
        """Test canceling timer when none exists."""
        app.unload_timer = None
        # Should not raise an exception
        app.cancel_unload_timer()
        assert app.unload_timer is None

    @patch("app.unload_model")
    def test_schedule_unload(self, mock_unload):
        """Test scheduling auto-unload."""
        app.schedule_unload()
        assert app.unload_timer is not None
        # Clean up
        app.cancel_unload_timer()


class TestLogoAndTheme:
    """Test logo and theme functions."""

    def test_get_logo_html_with_missing_file(self):
        """Test logo HTML when file is missing."""
        with patch.object(Path, "exists", return_value=False):
            result = app.get_logo_html()
            assert "PromptMill" in result
            assert "<h1" in result

    def test_create_theme(self):
        """Test theme creation."""
        theme = app.create_theme()
        assert theme is not None


class TestThreadSafety:
    """Test thread safety mechanisms."""

    def test_model_lock_exists(self):
        """Test that model lock is defined."""
        assert hasattr(app, "model_lock")
        assert app.model_lock is not None

    def test_model_lock_is_threading_lock(self):
        """Test that model lock is a proper threading lock."""
        import threading
        assert isinstance(app.model_lock, type(threading.Lock()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
