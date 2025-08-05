"""Core interfaces for live-crew orchestration components.

Protocols define clean interfaces for orchestration components,
enabling implementation swapping without breaking existing code.
This supports both simple and distributed deployment scenarios.
"""

from typing import Any, AsyncIterator, Protocol

from live_crew.core.models import Action, Event
from live_crew.core.dependencies import Dependency


class EventTransport(Protocol):
    """Protocol for event transport implementations.

    Supports both file-based and streaming transport implementations.
    """

    async def publish_event(self, event: Event[Any]) -> None:
        """Publish an event to the transport layer.

        Args:
            event: The event to publish
        """
        ...

    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        """Subscribe to events from the transport layer.

        Yields:
            Events as they become available
        """
        ...


class ActionTransport(Protocol):
    """Protocol for action transport implementations.

    Supports both console-based and streaming transport implementations.
    """

    async def publish_action(self, action: Action[Any]) -> None:
        """Publish an action to the transport layer.

        Args:
            action: The action to publish
        """
        ...

    async def subscribe_actions(self) -> AsyncIterator[Action[Any]]:
        """Subscribe to actions from the transport layer.

        Yields:
            Actions as they become available
        """
        ...


class ContextBackend(Protocol):
    """Protocol for context storage implementations.

    Supports both in-memory and distributed context backends.
    """

    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        """Get context snapshot for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index

        Returns:
            Context snapshot as a dictionary
        """
        ...

    async def apply_diff(
        self, stream_id: str, slice_idx: int, diff: dict[str, Any]
    ) -> None:
        """Apply a context diff for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index
            diff: The context diff to apply
        """
        ...

    async def clear_stream(self, stream_id: str) -> None:
        """Clear all context data for a stream.

        Args:
            stream_id: The stream identifier to clear
        """
        ...


class SchedulerBackend(Protocol):
    """Protocol for scheduler implementations.

    Supports both memory-based and distributed scheduling implementations.
    """

    async def schedule_crew(
        self, crew_id: str, slice_idx: int, dependencies: list[Dependency]
    ) -> None:
        """Schedule a crew for execution in a specific slice.

        Args:
            crew_id: The crew identifier
            slice_idx: The slice index to schedule in
            dependencies: List of dependencies that must be satisfied
        """
        ...

    async def mark_crew_complete(self, crew_id: str, slice_idx: int) -> None:
        """Mark a crew as completed for a specific slice.

        Args:
            crew_id: The crew identifier
            slice_idx: The slice index
        """
        ...

    async def get_pending_crews(self, slice_idx: int) -> list[str]:
        """Get list of crews pending execution for a slice.

        Args:
            slice_idx: The slice index to check

        Returns:
            List of crew IDs pending execution
        """
        ...


class EventHandler(Protocol):
    """Protocol for event handling implementations.

    Defines how crews process events into actions.
    """

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle an event and produce actions.

        Args:
            event: The event to process
            context: Current context snapshot

        Returns:
            List of actions produced from the event
        """
        ...

    @property
    def crew_id(self) -> str:
        """Get the crew identifier for this handler.

        Returns:
            The crew identifier
        """
        ...


class CrewRegistry(Protocol):
    """Protocol for crew registry implementations.

    Manages registration and retrieval of crews and their handlers.
    """

    def register_crew(
        self, handler: EventHandler, dependencies: list[Dependency]
    ) -> None:
        """Register a crew with its handler and dependencies.

        Args:
            handler: The event handler for the crew
            dependencies: List of dependencies for the crew
        """
        ...

    def get_handler(self, crew_id: str) -> EventHandler | None:
        """Get the event handler for a crew.

        Args:
            crew_id: The crew identifier

        Returns:
            The event handler or None if not found
        """
        ...

    def get_dependencies(self, crew_id: str) -> list[Dependency]:
        """Get the dependencies for a crew.

        Args:
            crew_id: The crew identifier

        Returns:
            List of dependencies for the crew
        """
        ...

    def list_crews(self) -> list[str]:
        """List all registered crew IDs.

        Returns:
            List of all registered crew IDs
        """
        ...
