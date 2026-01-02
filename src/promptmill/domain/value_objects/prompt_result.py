"""Prompt generation result value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptGenerationResult:
    """Value object for prompt generation output.

    Attributes:
        content: The generated prompt text.
        model_used: Name of the model that generated the prompt.
        role_used: Name of the role used for generation.
    """

    content: str
    model_used: str
    role_used: str

    @property
    def char_count(self) -> int:
        """Get character count of generated content.

        Returns:
            Number of characters in content.
        """
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get approximate word count of generated content.

        Returns:
            Number of whitespace-separated words.
        """
        return len(self.content.split())

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"PromptGenerationResult("
            f"chars={self.char_count}, "
            f"words={self.word_count}, "
            f"model={self.model_used!r})"
        )
