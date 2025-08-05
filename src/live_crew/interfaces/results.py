"""Result collection strategies for orchestration.

This module implements the Strategy pattern to cleanly separate
action publishing from result collection, eliminating duck typing
and providing clear interface contracts.
"""

from typing import Any, Protocol

from live_crew.core.models import Action


class ResultCollector(Protocol):
    """Protocol for collecting orchestration results.

    This interface separates result collection from action transport,
    allowing different strategies for different deployment scenarios.
    """

    async def get_actions(self) -> list[Action[Any]]:
        """Retrieve collected actions from processing.

        Returns:
            List of actions generated during orchestration
        """
        ...

    async def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with processing metrics like events_processed,
            time_slices, etc. Implementation-specific keys allowed.
        """
        ...

    def supports_collection(self) -> bool:
        """Check if this collector supports result gathering.

        Returns:
            True if get_actions() returns meaningful data,
            False if actions are streamed/discarded
        """
        ...

    def update_statistics(self, stats: dict[str, Any]) -> None:
        """Update processing statistics.

        Args:
            stats: Dictionary of statistics to update
        """
        ...


class CollectingResultCollector(ResultCollector):
    """Result collector for memory-based/testing scenarios.

    Works with action transports that store actions in memory
    for later retrieval (like CollectingActionTransport).
    """

    def __init__(self, collecting_transport):
        """Initialize with a transport that supports action collection.

        Args:
            collecting_transport: Transport with .actions attribute
        """
        self._transport = collecting_transport
        self._statistics: dict[str, Any] = {}

    async def get_actions(self) -> list[Action[Any]]:
        """Get actions from the collecting transport."""
        if hasattr(self._transport, "actions"):
            return self._transport.actions
        else:
            return []

    async def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return self._statistics.copy()

    def supports_collection(self) -> bool:
        """This collector supports action collection."""
        return True

    def update_statistics(self, stats: dict[str, Any]) -> None:
        """Update processing statistics.

        Args:
            stats: Dictionary of statistics to update
        """
        self._statistics.update(stats)


class NullResultCollector(ResultCollector):
    """Result collector for streaming/console scenarios.

    Used when actions are immediately streamed, printed, or sent
    to external systems without local storage.
    """

    def __init__(self):
        """Initialize null collector."""
        self._statistics: dict[str, Any] = {}

    async def get_actions(self) -> list[Action[Any]]:
        """Cannot collect from streaming transports."""
        return []

    async def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return self._statistics.copy()

    def supports_collection(self) -> bool:
        """This collector does not support action collection."""
        return False

    def update_statistics(self, stats: dict[str, Any]) -> None:
        """Update processing statistics.

        Args:
            stats: Dictionary of statistics to update
        """
        self._statistics.update(stats)


class SchedulerResultCollector(ResultCollector):
    """Result collector that gathers statistics from scheduler.

    This collector can work with any action transport by getting
    statistics directly from the scheduler component.
    """

    def __init__(self, scheduler, action_transport):
        """Initialize with scheduler and transport references.

        Args:
            scheduler: MemoryScheduler instance for statistics
            action_transport: Action transport (may or may not collect)
        """
        self._scheduler = scheduler
        self._action_transport = action_transport
        self._additional_stats: dict[str, Any] = {}

    async def get_actions(self) -> list[Action[Any]]:
        """Try to get actions from transport if it supports collection."""
        if hasattr(self._action_transport, "actions"):
            return self._action_transport.actions
        else:
            return []

    async def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics from scheduler and transport."""
        stats = self._additional_stats.copy()

        # Get scheduler statistics
        if hasattr(self._scheduler, "_processed_slices"):
            stats["time_slices"] = len(self._scheduler._processed_slices)

        return stats

    def supports_collection(self) -> bool:
        """Check if underlying transport supports collection."""
        return hasattr(self._action_transport, "actions")

    def update_statistics(self, stats: dict[str, Any]) -> None:
        """Update additional statistics.

        Args:
            stats: Dictionary of statistics to update
        """
        self._additional_stats.update(stats)
