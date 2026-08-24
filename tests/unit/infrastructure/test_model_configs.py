"""Tests for the VRAM model tiers."""

import pytest

from promptmill.infrastructure.config.model_configs import (
    MODEL_CONFIGS,
    MODEL_KEYS_ORDERED,
    get_all_models,
    get_model_by_key,
    get_model_by_name,
    get_model_names,
    select_model_by_vram,
)

GB = 1024


class TestTierIntegrity:
    """Structural guarantees about the tier table."""

    def test_ordering_covers_every_tier(self) -> None:
        """The display order lists each configured tier exactly once."""
        assert sorted(MODEL_KEYS_ORDERED) == sorted(MODEL_CONFIGS)

    def test_keys_match_their_entry(self) -> None:
        """A model's key matches the dict key it is filed under."""
        for key, model in MODEL_CONFIGS.items():
            assert model.key == key

    def test_names_are_unique(self) -> None:
        """Display names are the dropdown values, so they must not collide."""
        names = get_model_names()
        assert len(names) == len(set(names))

    def test_context_length_is_non_decreasing(self) -> None:
        """Context grows with the tier; a bigger card never gets less."""
        contexts = [m.context_length for m in get_all_models()]
        assert contexts == sorted(contexts)

    def test_every_tier_declares_a_chat_format(self) -> None:
        """A wrong or missing template yields garbage output, not an error."""
        assert all(m.chat_format for m in get_all_models())

    def test_top_tier_is_not_an_older_model(self) -> None:
        """Regression guard: the 24GB tier once ran Dolphin 2.9.4 while the
        16GB tier ran 3.0, making the top tier a downgrade."""
        assert MODEL_CONFIGS["24gb_vram"].repo_id == MODEL_CONFIGS["16gb_vram"].repo_id
        assert MODEL_CONFIGS["24gb_vram"].context_length > (
            MODEL_CONFIGS["16gb_vram"].context_length
        )

    def test_only_cpu_tier_runs_on_cpu(self) -> None:
        """Every GPU tier offloads all layers."""
        for model in get_all_models():
            expected = 0 if model.key == "cpu_only" else -1
            assert model.n_gpu_layers == expected


class TestLookup:
    """Tests for the lookup helpers."""

    def test_get_by_key(self) -> None:
        """A known key resolves."""
        assert get_model_by_key("cpu_only") is MODEL_CONFIGS["cpu_only"]

    def test_get_by_unknown_key(self) -> None:
        """An unknown key is None, not an exception."""
        assert get_model_by_key("128gb_vram") is None

    def test_get_by_name_roundtrip(self) -> None:
        """Every display name resolves back to its model."""
        for model in get_all_models():
            assert get_model_by_name(model.name) is model

    def test_get_by_unknown_name(self) -> None:
        """An unknown name is None."""
        assert get_model_by_name("Quantum Tier") is None


class TestVramSelection:
    """Tests for the VRAM-to-tier mapping."""

    @pytest.mark.parametrize(
        ("vram_gb", "expected_key"),
        [
            (0, "cpu_only"),
            (2, "cpu_only"),
            (3, "4gb_vram"),
            (4, "4gb_vram"),
            (5, "6gb_vram"),
            (6, "6gb_vram"),
            (7, "8gb_vram"),
            (8, "8gb_vram"),
            (10, "12gb_vram"),
            (12, "12gb_vram"),
            (14, "16gb_vram"),
            (16, "16gb_vram"),
            (20, "24gb_vram"),
            (24, "24gb_vram"),
            (32, "24gb_vram"),
            (80, "24gb_vram"),
        ],
    )
    def test_tier_boundaries(self, vram_gb: int, expected_key: str) -> None:
        """Each VRAM size maps to its documented tier."""
        assert select_model_by_vram(vram_gb * GB).key == expected_key

    def test_selection_is_monotonic(self) -> None:
        """More VRAM never selects a smaller tier."""
        rank = {key: i for i, key in enumerate(MODEL_KEYS_ORDERED)}
        ranks = [rank[select_model_by_vram(gb * GB).key] for gb in range(0, 40)]
        assert ranks == sorted(ranks)

    def test_zero_vram_is_cpu(self) -> None:
        """No GPU falls back to the CPU tier."""
        assert select_model_by_vram(0).n_gpu_layers == 0
