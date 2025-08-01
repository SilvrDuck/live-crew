"""File-based transport implementation for live-crew."""

import json
from pathlib import Path
from typing import Any, AsyncIterator

from live_crew.core.models import Event


class FileEventTransport:
    """File-based event transport for Slice 1.

    Reads events from a JSON file, suitable for batch processing
    and testing scenarios.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize file event transport.

        Args:
            file_path: Path to JSON file containing events
        """
        self.file_path = file_path

    async def publish_event(self, event: Event[Any]) -> None:
        """Publishing not supported for file-based transport.

        Args:
            event: The event to publish

        Raises:
            NotImplementedError: File transport is read-only
        """
        raise NotImplementedError("File transport is read-only")

    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        """Subscribe to events from JSON file.

        Yields:
            Events loaded from the JSON file

        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file contains invalid JSON or event data
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Event file not found: {self.file_path}")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support both single event and array of events
            if isinstance(data, dict):
                events_data = [data]
            elif isinstance(data, list):
                events_data = data
            else:
                raise ValueError(
                    "JSON file must contain an event object or array of events"
                )

            for event_data in events_data:
                try:
                    # Create Event instance with proper validation
                    event = Event[Any](**event_data)
                    yield event
                except Exception as e:
                    raise ValueError(f"Invalid event data: {e}") from e

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in event file: {e}") from e
