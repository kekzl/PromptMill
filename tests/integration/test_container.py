"""Integration tests for the DI container and full system initialization."""

import pytest

# Skip all tests if gradio is not installed
pytest.importorskip("gradio")

from promptmill.container import Container
from promptmill.domain.entities.role import RoleCategory
from promptmill.infrastructure.config.settings import Settings


class TestContainerIntegration:
    """Integration tests for Container initialization."""

    @pytest.fixture
    def container(self, tmp_path) -> Container:
        """Create a container with test settings."""
        settings = Settings(
            host="127.0.0.1",
            port=7610,
            models_dir=tmp_path / "models",
        )
        return Container(settings=settings)

    def test_container_initializes_all_components(self, container: Container) -> None:
        """Test that container initializes all components correctly."""
        # Verify core components
        assert container.role_repository is not None
        assert container.model_repository is not None
        assert container.gpu_detector is not None
        assert container.llm is not None

    def test_role_repository_has_all_roles(self, container: Container) -> None:
        """Test that all 102 roles are loaded."""
        assert container.role_repository.count() == 102

    def test_role_repository_has_all_categories(self, container: Container) -> None:
        """Test that all categories have roles."""
        categories = container.role_repository.get_categories()

        assert RoleCategory.VIDEO in categories
        assert RoleCategory.IMAGE in categories
        assert RoleCategory.AUDIO in categories
        assert RoleCategory.THREE_D in categories
        assert RoleCategory.CREATIVE in categories

    def test_role_repository_category_counts(self, container: Container) -> None:
        """Test role counts per category."""
        repo = container.role_repository

        assert repo.count_by_category(RoleCategory.VIDEO) == 22
        assert repo.count_by_category(RoleCategory.IMAGE) == 21
        assert repo.count_by_category(RoleCategory.AUDIO) == 13
        assert repo.count_by_category(RoleCategory.THREE_D) == 12
        assert repo.count_by_category(RoleCategory.CREATIVE) == 34

    def test_default_model_selection(self, container: Container) -> None:
        """Test that default model is selected based on GPU."""
        model = container.default_model

        assert model is not None
        assert model.name is not None
        assert model.repo_id is not None
        assert model.filename is not None

    def test_use_cases_are_wired(self, container: Container) -> None:
        """Test that use cases are properly wired."""
        # Get use cases from container
        assert container.generate_prompt_use_case is not None
        assert container.load_model_use_case is not None
        assert container.unload_model_use_case is not None
        assert container.select_model_use_case is not None

    def test_services_are_wired(self, container: Container) -> None:
        """Test that services are properly wired."""
        assert container.prompt_service is not None
        assert container.model_service is not None
        assert container.health_service is not None

    def test_health_service_returns_status(self, container: Container) -> None:
        """Test health service returns valid status."""
        status = container.health_service.get_status()

        assert status is not None
        assert "status" in status
        assert status["status"] in ("healthy", "degraded")
        assert "roles_count" in status
        assert status["roles_count"] == 102

    def test_container_shutdown(self, container: Container) -> None:
        """Test container shutdown doesn't raise."""
        # Should not raise
        container.shutdown()

    def test_model_service_lists_models(self, container: Container) -> None:
        """Test model service can list available models."""
        models = container.model_service.get_available_models()

        assert len(models) == 7  # 7 model tiers
        assert all(m.key for m in models)
        assert all(m.name for m in models)


class TestRoleIntegration:
    """Integration tests for role lookup across the system."""

    @pytest.fixture
    def container(self, tmp_path) -> Container:
        """Create a container with test settings."""
        settings = Settings(
            host="127.0.0.1",
            port=7610,
            models_dir=tmp_path / "models",
        )
        return Container(settings=settings)

    def test_lookup_role_by_display_name(self, container: Container) -> None:
        """Test looking up roles by display name."""
        repo = container.role_repository

        # Get all display names
        display_names = repo.get_display_names()
        assert len(display_names) == 102

        # Look up first role
        first_name = display_names[0]
        role = repo.get_by_display_name(first_name)

        assert role is not None
        assert role.display_name == first_name

    def test_all_roles_have_system_prompts(self, container: Container) -> None:
        """Test that all roles have non-empty system prompts."""
        roles = container.role_repository.get_all()

        for role in roles:
            assert role.system_prompt, f"Role {role.name} has empty system prompt"
            assert len(role.system_prompt) > 50, f"Role {role.name} has too short prompt"

    def test_video_roles_exist(self, container: Container) -> None:
        """Test that expected video roles exist."""
        video_roles = container.role_repository.get_by_category(RoleCategory.VIDEO)
        role_names = {r.name for r in video_roles}

        expected = {"Wan2.1", "Sora", "Runway Gen-3", "Kling", "Pika"}
        assert expected.issubset(role_names)

    def test_image_roles_exist(self, container: Container) -> None:
        """Test that expected image roles exist."""
        image_roles = container.role_repository.get_by_category(RoleCategory.IMAGE)
        role_names = {r.name for r in image_roles}

        expected = {"Midjourney", "FLUX", "DALL-E", "Stable Diffusion"}
        assert expected.issubset(role_names)
