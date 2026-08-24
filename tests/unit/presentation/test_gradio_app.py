"""Tests for the Gradio presentation layer.

These exercise the handler functions directly; no browser or Gradio server is
started.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("gradio")

from promptmill.domain.entities.model import Model
from promptmill.domain.entities.role import RoleCategory
from promptmill.infrastructure.adapters.role_repository_adapter import RoleRepositoryAdapter
from promptmill.presentation.examples import examples_for
from promptmill.presentation.gradio_app import ALL_CATEGORIES, GradioApp
from promptmill.presentation.history import HistoryEntry, labels


@pytest.fixture
def cpu_model() -> Model:
    """A model tier used as the default."""
    return Model(
        key="cpu_only",
        name="CPU Only (2-4GB RAM)",
        repo_id="test/repo",
        filename="test.gguf",
        context_length=4096,
        n_gpu_layers=0,
        description="Test tier",
        vram_required="~1.5GB",
    )


@pytest.fixture
def gpu_model() -> Model:
    """A second tier, used to check the GPU layer slider follows the model."""
    return Model(
        key="8gb_vram",
        name="8GB VRAM",
        repo_id="test/repo",
        filename="test-8b.gguf",
        context_length=8192,
        n_gpu_layers=-1,
        description="Bigger test tier",
        vram_required="~6GB",
    )


@pytest.fixture
def app(cpu_model: Model, gpu_model: Model) -> GradioApp:
    """A GradioApp wired to the real role data and mocked services."""
    by_name = {cpu_model.name: cpu_model, gpu_model.name: gpu_model}

    model_service = MagicMock()
    model_service.get_model_names.return_value = [cpu_model.name, gpu_model.name]
    model_service.get_model_by_name.side_effect = by_name.get

    return GradioApp(
        prompt_service=MagicMock(),
        model_service=model_service,
        health_service=MagicMock(),
        role_repository=RoleRepositoryAdapter(),
        assets_dir=Path("assets"),
        gpu_info=None,
        default_model=cpu_model,
    )


class TestCategoryFilter:
    """Tests for narrowing the target list."""

    def test_all_categories_returns_everything(self, app: GradioApp) -> None:
        """The "All" option does not filter."""
        assert len(app._role_choices_for(ALL_CATEGORIES)) == app.role_repository.count()

    def test_single_category_is_a_subset(self, app: GradioApp) -> None:
        """A category returns only its own targets."""
        choices = app._role_choices_for(RoleCategory.AUDIO.value)
        assert choices
        assert len(choices) < app.role_repository.count()
        assert all(c.startswith("[Audio] ") for c in choices)

    def test_every_category_yields_targets(self, app: GradioApp) -> None:
        """No category filter produces an empty dropdown."""
        for category in RoleCategory:
            assert app._role_choices_for(category.value)

    def test_category_change_selects_first_target(self, app: GradioApp) -> None:
        """Switching category picks a valid target rather than leaving a stale one."""
        dropdown, role_info, texts, *buttons = app._on_category_change(RoleCategory.THREE_D.value)
        assert dropdown["value"].startswith("[3D] ")
        assert role_info
        assert len(texts) == 6
        assert len(buttons) == 6

    def test_category_change_swaps_examples(self, app: GradioApp) -> None:
        """Examples follow the category, not the video defaults."""
        _, _, texts, *_ = app._on_category_change(RoleCategory.AUDIO.value)
        assert texts == [text for _, text in examples_for(RoleCategory.AUDIO)]


class TestRoleSelection:
    """Tests for target-driven UI updates."""

    def test_role_change_returns_description_and_examples(self, app: GradioApp) -> None:
        """Selecting a target refreshes its description and example set."""
        target = app._role_choices_for(RoleCategory.IMAGE.value)[0]
        role_info, texts, *buttons = app._on_role_change(target)
        assert role_info.startswith("**")
        assert texts == [text for _, text in examples_for(RoleCategory.IMAGE)]
        assert len(buttons) == 6

    def test_unknown_role_falls_back_to_creative(self, app: GradioApp) -> None:
        """An unknown target does not raise."""
        role_info, texts, *_ = app._on_role_change("[Nope] Missing")
        assert role_info == ""
        assert texts == [text for _, text in examples_for(RoleCategory.CREATIVE)]

    def test_example_text_lookup(self) -> None:
        """Example buttons read their text out of session state."""
        assert GradioApp._example_text(["a", "b", "c"], 1) == "b"

    def test_example_text_out_of_range(self) -> None:
        """A stale state does not crash the button."""
        assert GradioApp._example_text([], 4) == ""


class TestModelSelection:
    """Tests for the model dropdown side effects."""

    def test_model_change_updates_gpu_layers(self, app: GradioApp, gpu_model: Model) -> None:
        """The GPU layer slider follows the selected tier."""
        info, slider = app._on_model_change(gpu_model.name)
        assert gpu_model.description in info
        assert slider["value"] == gpu_model.n_gpu_layers

    def test_model_info_shows_context_length(self, app: GradioApp, cpu_model: Model) -> None:
        """Context length is visible, since it drives VRAM use."""
        assert "4,096" in app._get_model_info(cpu_model.name)

    def test_unknown_model_change_is_safe(self, app: GradioApp) -> None:
        """An unknown model name does not raise."""
        info, _ = app._on_model_change("does not exist")
        assert "not found" in info.lower()


class TestGeneration:
    """Tests for generation input handling and history recording."""

    def test_empty_input_short_circuits(self, app: GradioApp) -> None:
        """Blank input never reaches the model."""
        out = list(
            app._generate_prompt("   ", "[Video] Sora 2", "CPU Only (2-4GB RAM)", 0.7, 256, 0)
        )
        assert out == ["Please enter an idea or description to generate a prompt."]
        app.prompt_service.generate.assert_not_called()

    def test_gpu_layers_reach_the_service(self, app: GradioApp) -> None:
        """The GPU layers slider is wired through, not ignored."""
        app.prompt_service.generate.return_value = iter(["ok"])
        list(app._generate_prompt("idea", "[Video] Sora 2", "CPU Only (2-4GB RAM)", 0.7, 256, 17))
        assert app.prompt_service.generate.call_args.kwargs["n_gpu_layers_override"] == 17

    def test_unknown_model_reports_error(self, app: GradioApp) -> None:
        """A missing model yields a message instead of an exception."""
        out = list(app._generate_prompt("idea", "[Video] Sora 2", "nope", 0.7, 256, 0))
        assert out == ["Model not found: nope"]

    def test_invalid_request_is_reported(self, app: GradioApp) -> None:
        """Validation errors surface as text, not a traceback."""
        out = list(
            app._generate_prompt("idea", "[Video] Sora 2", "CPU Only (2-4GB RAM)", 9.0, 256, 0)
        )
        assert out[0].startswith("Error:")

    def test_record_generation_appends_and_labels(self, app: GradioApp) -> None:
        """A finished generation lands in history with a dropdown label."""
        history, dropdown, stats = app._record_generation(
            [], "my idea", "[Video] Sora 2", "CPU Only (2-4GB RAM)", "a generated prompt"
        )
        assert len(history) == 1
        assert dropdown["choices"] == labels(history)
        assert "18 characters" in stats
        assert "3 words" in stats

    def test_record_generation_skips_empty_output(self, app: GradioApp) -> None:
        """Nothing is recorded when generation produced nothing."""
        history, _, stats = app._record_generation([], "my idea", "[Video] Sora 2", "m", "   ")
        assert history == []
        assert stats == ""

    def test_restore_history_returns_stored_values(self, app: GradioApp) -> None:
        """Restoring puts the stored idea, target and prompt back."""
        history, _, _ = app._record_generation(
            [], "my idea", "[Video] Sora 2", "CPU Only (2-4GB RAM)", "a generated prompt"
        )
        idea, role, output, stats = app._restore_history(history, labels(history)[0])
        assert idea == "my idea"
        assert role["value"] == "[Video] Sora 2"
        assert output == "a generated prompt"
        assert stats

    def test_restore_unknown_label_is_noop(self, app: GradioApp) -> None:
        """Restoring nothing leaves the UI alone."""
        _, _, _, stats = app._restore_history([], "1. missing")
        assert stats == ""

    def test_clear_history(self) -> None:
        """Clearing empties both the state and the dropdown."""
        history, dropdown = GradioApp._clear_history()
        assert history == []
        assert dropdown["choices"] == []


class TestBlocksConstruction:
    """The UI must actually build."""

    def test_create_builds_blocks(self, app: GradioApp) -> None:
        """Blocks construction wires every handler without error."""
        blocks = app.create()
        assert blocks is not None
        assert app._app is blocks


class TestHistoryEntryType:
    """Guard the history payload shape used by the Gradio state."""

    def test_recorded_entry_is_history_entry(self, app: GradioApp) -> None:
        """State holds typed entries, not raw tuples."""
        history, _, _ = app._record_generation([], "idea", "[Video] Sora 2", "m", "prompt")
        assert isinstance(history[0], HistoryEntry)
