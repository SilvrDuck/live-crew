"""Event handler implementations for live-crew orchestration."""

from typing import Any

from live_crew.core.models import Action, Event


class EchoEventHandler:
    """Simple echo event handler for testing and demonstration.

    Converts each event into an action with the same payload,
    adding metadata about the processing.
    """

    def __init__(self, crew_id: str) -> None:
        """Initialize echo handler with crew ID.

        Args:
            crew_id: The crew identifier for this handler
        """
        self._crew_id = crew_id

    @property
    def crew_id(self) -> str:
        """Get the crew identifier for this handler.

        Returns:
            The crew identifier
        """
        return self._crew_id

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle an event by echoing it as an action.

        Args:
            event: The event to process
            context: Current context snapshot (unused in echo)

        Returns:
            List containing single action that echoes the event
        """
        # Create echo action with metadata
        echo_payload = {
            "echo_of": event.kind,
            "original_payload": event.payload,
            "processed_by": self._crew_id,
            "context_keys": list(context.keys()) if context else [],
        }

        action = Action[dict[str, Any]](
            ts=event.ts,  # Use same timestamp
            kind=f"echo_{event.kind}",
            stream_id=event.stream_id,
            payload=echo_payload,
            ttl_ms=5000,  # 5 second TTL for echo actions
        )

        return [action]
