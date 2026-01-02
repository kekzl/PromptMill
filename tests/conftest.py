"""Pytest configuration and fixtures for PromptMill tests."""

import sys
from pathlib import Path
from threading import RLock
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from promptmill.domain.entities.gpu_info import GPUInfo
from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import Role, RoleCategory
from promptmill.domain.ports.gpu_detector_port import GPUDetectorPort
from promptmill.domain.ports.llm_port import LLMPort
from promptmill.domain.ports.model_repository_port import ModelRepositoryPort
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort
from promptmill.infrastructure.config.settings import Settings

# =============================================================================
# Domain Fixtures
# =============================================================================


@pytest.fixture
def sample_model() -> Model:
    """Provide a sample Model entity."""
    return Model(
        key="test_model",
        name="Test Model (8GB VRAM)",
        repo_id="test/model-repo",
        filename="test-model.gguf",
        context_length=4096,
        n_gpu_layers=-1,
        description="Test model for unit tests",
        vram_required="~4GB",
    )


@pytest.fixture
def cpu_model() -> Model:
    """Provide a CPU-only Model entity."""
    return Model(
        key="cpu_only",
        name="CPU Only (2-4GB RAM)",
        repo_id="test/cpu-model",
        filename="cpu-model.gguf",
        context_length=2048,
        n_gpu_layers=0,
        description="CPU-only model",
        vram_required="~1GB",
    )


@pytest.fixture
def sample_role() -> Role:
    """Provide a sample Role entity."""
    return Role(
        name="TestRole",
        category=RoleCategory.VIDEO,
        description="A test role for unit tests",
        system_prompt="You are a test assistant. Generate test prompts.",
    )


@pytest.fixture
def sample_gpu_info() -> GPUInfo:
    """Provide sample GPU information."""
    return GPUInfo(
        name="NVIDIA GeForce RTX 4090",
        vram_mb=24576,
        driver_version="535.154.05",
    )


@pytest.fixture
def cpu_only_gpu_info() -> GPUInfo:
    """Provide CPU-only GPU information."""
    return GPUInfo.cpu_only()


# =============================================================================
# Mock Port Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLMPort implementation."""
    mock = MagicMock(spec=LLMPort)
    mock.is_loaded.return_value = False
    mock.get_loaded_model_path.return_value = None

    def generate_mock(*_args, **_kwargs):
        yield "Test "
        yield "response "
        yield "content"

    mock.generate.side_effect = generate_mock
    return mock


@pytest.fixture
def mock_model_repository(tmp_path: Path) -> MagicMock:
    """Create a mock ModelRepositoryPort implementation."""
    mock = MagicMock(spec=ModelRepositoryPort)
    mock.get_model_path.return_value = None
    mock.download_model.return_value = tmp_path / "test-model.gguf"
    mock.delete_model.return_value = True
    mock.get_disk_usage.return_value = 0
    mock.get_available_space.return_value = 100 * 1024 * 1024 * 1024  # 100GB
    return mock


@pytest.fixture
def mock_gpu_detector() -> MagicMock:
    """Create a mock GPUDetectorPort implementation."""
    mock = MagicMock(spec=GPUDetectorPort)
    mock.detect.return_value = GPUInfo(
        name="NVIDIA GeForce RTX 4090",
        vram_mb=24576,
        driver_version="535.154.05",
    )
    mock.is_cuda_available.return_value = True
    return mock


@pytest.fixture
def mock_gpu_detector_no_gpu() -> MagicMock:
    """Create a mock GPUDetectorPort with no GPU."""
    mock = MagicMock(spec=GPUDetectorPort)
    mock.detect.return_value = None
    mock.is_cuda_available.return_value = False
    return mock


@pytest.fixture
def mock_role_repository(sample_role: Role) -> MagicMock:
    """Create a mock RoleRepositoryPort implementation."""
    mock = MagicMock(spec=RoleRepositoryPort)
    mock.get_all.return_value = [sample_role]
    mock.get_by_display_name.return_value = sample_role
    mock.get_by_name.return_value = sample_role
    mock.get_by_category.return_value = [sample_role]
    mock.get_display_names.return_value = [sample_role.display_name]
    mock.get_categories.return_value = [RoleCategory.VIDEO]
    mock.count.return_value = 1
    mock.count_by_category.return_value = 1
    return mock


# =============================================================================
# Infrastructure Fixtures
# =============================================================================


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary directory."""
    return Settings(
        host="127.0.0.1",
        port=7610,
        models_dir=tmp_path / "models",
        default_batch_size=512,
        default_chat_format="llama-3",
        gpu_detection_timeout=5,
        max_prompt_length=10000,
        min_temperature=0.1,
        max_temperature=2.0,
        min_tokens=100,
        max_tokens=2000,
        unload_delay_seconds=10,
    )


@pytest.fixture
def model_lock() -> RLock:
    """Provide a threading lock for tests."""
    return RLock()


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def mock_model_file(tmp_path: Path) -> Path:
    """Create a mock model file."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "test-model.gguf"
    model_file.write_bytes(b"mock model data" * 1000)
    return model_file


# =============================================================================
# Patching Fixtures
# =============================================================================


@pytest.fixture
def _patch_llama():
    """Patch the Llama class from llama_cpp (lazily imported).

    Creates a mock llama_cpp module since it may not be installed.
    """
    import sys

    # Create mock module
    mock_llama_cpp = MagicMock()
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = iter([
        {"choices": [{"delta": {"content": "Test "}}]},
        {"choices": [{"delta": {"content": "response"}}]},
    ])
    mock_llama_cpp.Llama.return_value = mock_instance

    # Inject mock module
    sys.modules["llama_cpp"] = mock_llama_cpp

    try:
        yield mock_llama_cpp.Llama
    finally:
        # Clean up
        if "llama_cpp" in sys.modules:
            del sys.modules["llama_cpp"]


@pytest.fixture
def patch_hf_download(tmp_path: Path):
    """Patch HuggingFace Hub download."""
    with patch("promptmill.infrastructure.adapters.huggingface_adapter.hf_hub_download") as mock:
        mock.return_value = str(tmp_path / "test-model.gguf")
        yield mock
