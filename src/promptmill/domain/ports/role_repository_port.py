"""Role repository port (interface)."""

from abc import ABC, abstractmethod

from promptmill.domain.entities.role import Role, RoleCategory


class RoleRepositoryPort(ABC):
    """Port for role/template retrieval operations.

    This defines the interface for accessing the role templates
    used for prompt generation.
    """

    @abstractmethod
    def get_all(self) -> list[Role]:
        """Get all available roles.

        Returns:
            List of all Role objects, sorted by category then name.
        """
        ...

    @abstractmethod
    def get_by_display_name(self, display_name: str) -> Role | None:
        """Get role by its display name.

        Args:
            display_name: Display name in "[Category] Name" format.

        Returns:
            Role if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_by_name(self, name: str) -> Role | None:
        """Get role by its name (without category prefix).

        Args:
            name: Role name (e.g., "Midjourney").

        Returns:
            Role if found (first match if duplicates), None otherwise.
        """
        ...

    @abstractmethod
    def get_by_category(self, category: RoleCategory) -> list[Role]:
        """Get all roles in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of roles in the specified category, sorted by name.
        """
        ...

    @abstractmethod
    def get_display_names(self) -> list[str]:
        """Get all role display names.

        Returns:
            List of display names in "[Category] Name" format.
        """
        ...

    @abstractmethod
    def get_categories(self) -> list[RoleCategory]:
        """Get all available categories.

        Returns:
            List of categories that have at least one role.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Get total number of roles.

        Returns:
            Total count of available roles.
        """
        ...

    @abstractmethod
    def count_by_category(self, category: RoleCategory) -> int:
        """Get number of roles in a category.

        Args:
            category: The category to count.

        Returns:
            Number of roles in the category.
        """
        ...
