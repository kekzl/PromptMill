"""Role entity representing a prompt generation role/template."""

from dataclasses import dataclass
from enum import StrEnum


class RoleCategory(StrEnum):
    """Categories of prompt generation roles."""

    VIDEO = "Video"
    IMAGE = "Image"
    AUDIO = "Audio"
    THREE_D = "3D"
    CREATIVE = "Creative"

    @classmethod
    def from_string(cls, value: str) -> "RoleCategory":
        """Create category from string, with fallback to CREATIVE.

        Args:
            value: Category string to parse.

        Returns:
            Matching RoleCategory or CREATIVE as default.
        """
        for category in cls:
            if category.value.lower() == value.lower():
                return category
        return cls.CREATIVE


@dataclass(frozen=True, slots=True)
class Role:
    """Domain entity representing a prompt generation role.

    A role defines the system prompt and context for a specific
    AI generation target (e.g., Midjourney, Suno, etc.).

    Attributes:
        name: Role identifier (e.g., "Midjourney").
        category: The category this role belongs to.
        description: Short description of the role's purpose.
        system_prompt: The system prompt template for this role.
    """

    name: str
    category: RoleCategory
    description: str
    system_prompt: str

    @property
    def display_name(self) -> str:
        """Get formatted display name with category prefix.

        Returns:
            Display name in format "[Category] Name".
        """
        return f"[{self.category.value}] {self.name}"

    @classmethod
    def parse_display_name(cls, display_name: str) -> tuple[str, str]:
        """Parse a display name into category and name components.

        Args:
            display_name: Display name in "[Category] Name" format.

        Returns:
            Tuple of (category_str, name).

        Raises:
            ValueError: If display name format is invalid.
        """
        if not display_name.startswith("["):
            raise ValueError(f"Invalid display name format: {display_name}")

        try:
            bracket_end = display_name.index("]")
            category_str = display_name[1:bracket_end]
            name = display_name[bracket_end + 2 :]  # Skip "] "
            return category_str, name
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid display name format: {display_name}") from e
