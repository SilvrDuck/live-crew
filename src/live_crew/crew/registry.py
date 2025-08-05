"""Crew registry implementations for live-crew orchestration."""

from live_crew.core.dependencies import Dependency
from live_crew.interfaces.protocols import CrewRegistry, EventHandler


class SimpleCrewRegistry(CrewRegistry):
    """Simple in-memory crew registry implementation.

    Suitable for single-crew scenarios and testing.
    Can be extended or replaced for distributed deployment scenarios.
    """

    def __init__(self) -> None:
        """Initialize empty crew registry."""
        self._handlers: dict[str, EventHandler] = {}
        self._dependencies: dict[str, list[Dependency]] = {}

    def register_crew(
        self, handler: EventHandler, dependencies: list[Dependency]
    ) -> None:
        """Register a crew with its handler and dependencies.

        Args:
            handler: The event handler for the crew
            dependencies: List of dependencies for the crew

        Raises:
            ValueError: If crew_id is already registered
        """
        crew_id = handler.crew_id

        if crew_id in self._handlers:
            raise ValueError(f"Crew '{crew_id}' is already registered")

        self._handlers[crew_id] = handler
        self._dependencies[crew_id] = dependencies.copy()

    def get_handler(self, crew_id: str) -> EventHandler | None:
        """Get the event handler for a crew.

        Args:
            crew_id: The crew identifier

        Returns:
            The event handler or None if not found
        """
        return self._handlers.get(crew_id)

    def get_dependencies(self, crew_id: str) -> list[Dependency]:
        """Get the dependencies for a crew.

        Args:
            crew_id: The crew identifier

        Returns:
            List of dependencies for the crew (empty if crew not found)
        """
        return self._dependencies.get(crew_id, []).copy()

    def list_crews(self) -> list[str]:
        """List all registered crew IDs.

        Returns:
            List of all registered crew IDs
        """
        return list(self._handlers.keys())

    def clear(self) -> None:
        """Clear all registered crews."""
        self._handlers.clear()
        self._dependencies.clear()
