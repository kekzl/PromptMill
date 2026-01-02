"""Prompt generation request value object."""

from dataclasses import dataclass

# Validation constants
MAX_INPUT_LENGTH = 10000
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0
MIN_TOKENS = 100
MAX_TOKENS = 2000


@dataclass(frozen=True, slots=True)
class PromptGenerationRequest:
    """Value object for prompt generation input.

    This is an immutable request object that validates all inputs
    at construction time.

    Attributes:
        user_input: The user's description or idea to transform.
        role_display_name: Display name of the role to use (e.g., "[Video] Midjourney").
        temperature: LLM temperature (0.1-2.0).
        max_tokens: Maximum tokens to generate (100-2000).
    """

    user_input: str
    role_display_name: str
    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        # Validate user input
        if not self.user_input or not self.user_input.strip():
            raise ValueError("User input cannot be empty")

        if len(self.user_input) > MAX_INPUT_LENGTH:
            raise ValueError(f"User input exceeds maximum length of {MAX_INPUT_LENGTH} characters")

        # Validate role
        if not self.role_display_name:
            raise ValueError("Role display name cannot be empty")

        # Validate temperature
        if not MIN_TEMPERATURE <= self.temperature <= MAX_TEMPERATURE:
            raise ValueError(
                f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
                f"got {self.temperature}"
            )

        # Validate max_tokens
        if not MIN_TOKENS <= self.max_tokens <= MAX_TOKENS:
            raise ValueError(
                f"Max tokens must be between {MIN_TOKENS} and {MAX_TOKENS}, got {self.max_tokens}"
            )

    @property
    def stripped_input(self) -> str:
        """Get whitespace-stripped user input.

        Returns:
            User input with leading/trailing whitespace removed.
        """
        return self.user_input.strip()
