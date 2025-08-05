"""Context backend implementations for live-crew."""

from typing import Any

from live_crew.interfaces.protocols import ContextBackend


class DictContextBackend(ContextBackend):
    """Simple dictionary-based context backend.

    Stores context in memory using nested dictionaries.
    Suitable for single-process scenarios and testing.
    """

    def __init__(self) -> None:
        """Initialize empty context storage."""
        # Structure: {stream_id: {slice_idx: context_dict}}
        self._contexts: dict[str, dict[int, dict[str, Any]]] = {}

    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        """Get context snapshot for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index

        Returns:
            Context snapshot as a dictionary (empty if not exists)
        """
        return self._contexts.get(stream_id, {}).get(slice_idx, {}).copy()

    async def apply_diff(
        self, stream_id: str, slice_idx: int, diff: dict[str, Any]
    ) -> None:
        """Apply a context diff for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index
            diff: The context diff to apply (simple key-value updates)
        """
        # Ensure stream exists
        if stream_id not in self._contexts:
            self._contexts[stream_id] = {}

        # Ensure slice exists
        if slice_idx not in self._contexts[stream_id]:
            self._contexts[stream_id][slice_idx] = {}

        # Apply diff as simple key-value updates
        # In distributed scenarios, this would use proper diff-merge logic
        self._contexts[stream_id][slice_idx].update(diff)

    async def clear_stream(self, stream_id: str) -> None:
        """Clear all context data for a stream.

        Args:
            stream_id: The stream identifier to clear
        """
        if stream_id in self._contexts:
            del self._contexts[stream_id]

    def get_all_contexts(self) -> dict[str, dict[int, dict[str, Any]]]:
        """Get all contexts for debugging/testing.

        Returns:
            Complete context storage structure
        """
        return self._contexts.copy()

    def clear_all(self) -> None:
        """Clear all context data."""
        self._contexts.clear()
