"""Console-based transport implementation for live-crew."""

import json
from typing import Any, AsyncIterator

from live_crew.core.models import Action
from live_crew.interfaces.protocols import ActionTransport


class ConsoleActionTransport(ActionTransport):
    """Console-based action transport implementation.

    Prints actions to console with formatting, suitable for
    development and testing scenarios.
    """

    def __init__(self, format_json: bool = True) -> None:
        """Initialize console action transport.

        Args:
            format_json: Whether to pretty-print JSON output
        """
        self.format_json = format_json

    async def publish_action(self, action: Action[Any]) -> None:
        """Publish an action to console output.

        Args:
            action: The action to publish
        """
        if self.format_json:
            # Pretty-print the action as JSON
            action_dict = {
                "ts": action.ts.isoformat(),
                "kind": action.kind,
                "stream_id": action.stream_id,
                "payload": action.payload,
                "ttl_ms": action.ttl_ms,
            }
            print(json.dumps(action_dict, indent=2, default=str))
        else:
            # Simple string representation
            print(
                f"Action[{action.kind}]: {action.payload} (stream: {action.stream_id})"
            )

    async def subscribe_actions(self) -> AsyncIterator[Action[Any]]:
        """Subscription not supported for console transport.

        Yields:
            Never yields - console transport is write-only

        Raises:
            NotImplementedError: Console transport is write-only
        """
        raise NotImplementedError("Console transport is write-only")
        # This is to satisfy the AsyncIterator return type
        yield  # type: ignore # pragma: no cover
