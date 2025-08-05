"""File-based transport implementation for live-crew."""

import json
from pathlib import Path
from typing import Any, AsyncIterator

from live_crew.core.models import Event
from live_crew.interfaces.protocols import EventTransport
from live_crew.security.path_validation import validate_file_path


class FileEventTransport(EventTransport):
    """File-based event transport implementation.

    Reads events from a JSON file, suitable for batch processing
    and testing scenarios.
    """

    def __init__(
        self,
        file_path: Path,
        allowed_directories: list[Path] | None = None,
        allow_home_access: bool = False,
    ) -> None:
        """Initialize file event transport.

        Args:
            file_path: Path to JSON file containing events
            allowed_directories: List of directories allowed for file access.
                               If None, defaults to current working directory and subdirectories.
            allow_home_access: Whether to allow access to user home directories

        Raises:
            PathSecurityError: If file path fails security validation
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        # Validate path security before storing
        self.file_path = validate_file_path(
            file_path,
            allowed_directories=allowed_directories,
            allow_home_access=allow_home_access,
        )

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
            ValueError: If the file contains invalid JSON or event data

        Note:
            File existence and permissions are validated during initialization,
            so we can safely open the file here.
        """
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

    async def read_events(self) -> list[Event[Any]]:
        """Read all events from the JSON file as a list.

        Returns:
            List of all events from the file

        Raises:
            ValueError: If the file contains invalid JSON or event data

        Note:
            File existence and permissions are validated during initialization.
        """
        events = []
        async for event in self.subscribe_events():
            events.append(event)
        return events
