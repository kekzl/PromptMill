"""Tests for the per-session prompt history."""

from promptmill.domain.value_objects.prompt_result import PromptGenerationResult
from promptmill.presentation.history import (
    LABEL_INPUT_CHARS,
    MAX_HISTORY_ENTRIES,
    HistoryEntry,
    add_entry,
    find_by_label,
    labels,
)


def make_entry(text: str = "a lighthouse in a storm") -> HistoryEntry:
    """Build a history entry for tests."""
    return HistoryEntry(
        user_input=text,
        role_display_name="[Video] Sora 2",
        model_name="CPU Only (2-4GB RAM)",
        result=PromptGenerationResult(
            content="generated prompt text",
            model_used="CPU Only (2-4GB RAM)",
            role_used="[Video] Sora 2",
        ),
    )


class TestHistoryEntry:
    """Tests for HistoryEntry labels."""

    def test_label_contains_index_and_role(self) -> None:
        """Label carries position and target."""
        label = make_entry().label(1)
        assert label.startswith("1. [Video] Sora 2 - ")

    def test_label_truncates_long_input(self) -> None:
        """Long inputs are cut to the label budget."""
        label = make_entry("x" * 200).label(3)
        excerpt = label.split(" - ", 1)[1]
        assert len(excerpt) == LABEL_INPUT_CHARS
        assert excerpt.endswith("…")

    def test_label_collapses_whitespace(self) -> None:
        """Newlines in the input do not break the dropdown label."""
        label = make_entry("first line\n\n   second line").label(1)
        assert "first line second line" in label


class TestAddEntry:
    """Tests for add_entry."""

    def test_newest_first(self) -> None:
        """The most recent entry lands at the front."""
        history = add_entry([], make_entry("old"))
        history = add_entry(history, make_entry("new"))
        assert history[0].user_input == "new"
        assert history[1].user_input == "old"

    def test_does_not_mutate_input(self) -> None:
        """Gradio state must be replaced, not mutated in place."""
        original: list[HistoryEntry] = []
        add_entry(original, make_entry())
        assert original == []

    def test_caps_at_retention_limit(self) -> None:
        """History stops growing at the cap."""
        history: list[HistoryEntry] = []
        for i in range(MAX_HISTORY_ENTRIES + 5):
            history = add_entry(history, make_entry(f"idea {i}"))
        assert len(history) == MAX_HISTORY_ENTRIES
        assert history[0].user_input == f"idea {MAX_HISTORY_ENTRIES + 4}"


class TestLookup:
    """Tests for labels and find_by_label."""

    def test_labels_are_one_based(self) -> None:
        """Labels count from 1."""
        history = [make_entry("a"), make_entry("b")]
        assert labels(history)[0].startswith("1. ")
        assert labels(history)[1].startswith("2. ")

    def test_find_by_label_roundtrip(self) -> None:
        """A rendered label resolves back to its entry."""
        history = [make_entry("first"), make_entry("second")]
        found = find_by_label(history, labels(history)[1])
        assert found is not None
        assert found.user_input == "second"

    def test_find_by_unknown_label(self) -> None:
        """An unknown label yields None instead of raising."""
        assert find_by_label([make_entry()], "99. nope") is None

    def test_find_in_empty_history(self) -> None:
        """Empty history yields None."""
        assert find_by_label([], "1. anything") is None
