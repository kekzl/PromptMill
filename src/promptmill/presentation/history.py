"""Per-session prompt history.

History lives in a Gradio ``State``, so it is scoped to one browser session and
never shared between users or persisted to disk.
"""

from dataclasses import dataclass

from promptmill.domain.value_objects.prompt_result import PromptGenerationResult

# Keeping every generation would grow the session payload without bound.
MAX_HISTORY_ENTRIES = 20

# Longest input excerpt used in a history label.
LABEL_INPUT_CHARS = 45


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One generated prompt plus the inputs that produced it.

    Attributes:
        user_input: The idea the user typed.
        role_display_name: Target role in "[Category] Name" format.
        model_name: Display name of the model that generated the prompt.
        result: The generated prompt with its char/word counts.
    """

    user_input: str
    role_display_name: str
    model_name: str
    result: PromptGenerationResult

    def label(self, index: int) -> str:
        """Build the dropdown label for this entry.

        Args:
            index: 1-based position in the history list.

        Returns:
            Label in "N. [Category] Name - excerpt" form.
        """
        excerpt = " ".join(self.user_input.split())
        if len(excerpt) > LABEL_INPUT_CHARS:
            excerpt = excerpt[: LABEL_INPUT_CHARS - 1] + "…"
        return f"{index}. {self.role_display_name} - {excerpt}"


def add_entry(history: list[HistoryEntry], entry: HistoryEntry) -> list[HistoryEntry]:
    """Prepend an entry and trim to the retention limit.

    Args:
        history: Existing history, newest first.
        entry: The entry to add.

    Returns:
        A new list, newest first, capped at MAX_HISTORY_ENTRIES.
    """
    return [entry, *history][:MAX_HISTORY_ENTRIES]


def labels(history: list[HistoryEntry]) -> list[str]:
    """Build dropdown labels for a history list.

    Args:
        history: History entries, newest first.

    Returns:
        Labels in list order.
    """
    return [entry.label(i) for i, entry in enumerate(history, start=1)]


def find_by_label(history: list[HistoryEntry], label: str) -> HistoryEntry | None:
    """Look up an entry by its rendered label.

    Args:
        history: History entries, newest first.
        label: Label as produced by :func:`labels`.

    Returns:
        The matching entry, or None if the label is unknown.
    """
    for i, entry in enumerate(history, start=1):
        if entry.label(i) == label:
            return entry
    return None
