"""Role repository adapter implementing RoleRepositoryPort."""

import logging
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing import TypeVar

    def override(func: TypeVar("F")) -> TypeVar("F"):  # type: ignore[misc]
        return func  # type: ignore[return-value]

from promptmill.domain.entities.role import Role, RoleCategory
from promptmill.domain.ports.role_repository_port import RoleRepositoryPort
from promptmill.infrastructure.persistence.roles_data import ROLES_DATA

logger = logging.getLogger(__name__)


class RoleRepositoryAdapter(RoleRepositoryPort):
    """Adapter for role/template retrieval from static data.

    This adapter implements the RoleRepositoryPort interface using
    the predefined role data.
    """

    __slots__ = ("_by_category", "_by_display_name", "_by_name", "_roles")

    def __init__(self) -> None:
        """Initialize the adapter and build indexes."""
        self._roles: list[Role] = []
        self._by_display_name: dict[str, Role] = {}
        self._by_name: dict[str, Role] = {}
        self._by_category: dict[RoleCategory, list[Role]] = {
            cat: [] for cat in RoleCategory
        }

        self._load_roles()

    def _load_roles(self) -> None:
        """Load and index all roles from data."""
        for role_key, role_data in ROLES_DATA.items():
            category = RoleCategory.from_string(role_data["category"])
            role = Role(
                name=role_key,
                category=category,
                description=role_data.get("description", ""),
                system_prompt=role_data["system_prompt"],
            )

            self._roles.append(role)
            self._by_display_name[role.display_name] = role
            self._by_name[role.name] = role
            self._by_category[category].append(role)

        # Sort roles
        self._roles.sort(key=lambda r: (r.category.value, r.name))
        for cat_roles in self._by_category.values():
            cat_roles.sort(key=lambda r: r.name)

        logger.info(f"Loaded {len(self._roles)} roles")

    @override
    def get_all(self) -> list[Role]:
        """Get all available roles.

        Returns:
            List of all Role objects, sorted by category then name.
        """
        return list(self._roles)

    @override
    def get_by_display_name(self, display_name: str) -> Role | None:
        """Get role by its display name.

        Args:
            display_name: Display name in "[Category] Name" format.

        Returns:
            Role if found, None otherwise.
        """
        return self._by_display_name.get(display_name)

    @override
    def get_by_name(self, name: str) -> Role | None:
        """Get role by its name (without category prefix).

        Args:
            name: Role name (e.g., "Midjourney").

        Returns:
            Role if found, None otherwise.
        """
        return self._by_name.get(name)

    @override
    def get_by_category(self, category: RoleCategory) -> list[Role]:
        """Get all roles in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of roles in the specified category, sorted by name.
        """
        return list(self._by_category.get(category, []))

    @override
    def get_display_names(self) -> list[str]:
        """Get all role display names.

        Returns:
            List of display names in "[Category] Name" format.
        """
        return [role.display_name for role in self._roles]

    @override
    def get_categories(self) -> list[RoleCategory]:
        """Get all available categories.

        Returns:
            List of categories that have at least one role.
        """
        return [
            cat for cat in RoleCategory
            if self._by_category.get(cat)
        ]

    @override
    def count(self) -> int:
        """Get total number of roles.

        Returns:
            Total count of available roles.
        """
        return len(self._roles)

    @override
    def count_by_category(self, category: RoleCategory) -> int:
        """Get number of roles in a category.

        Args:
            category: The category to count.

        Returns:
            Number of roles in the category.
        """
        return len(self._by_category.get(category, []))
