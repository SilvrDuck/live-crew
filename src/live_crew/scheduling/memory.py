"""Memory-based scheduler implementation for live-crew.

Focused on time slicing and event distribution. Delegates complex orchestration
to specialized orchestrator components for clean separation of concerns.
Designed to be replaceable with distributed scheduler implementations.
"""

from datetime import datetime
from typing import Any

from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import LiveCrewConfig
from live_crew.core.dependencies import Dependency
from live_crew.core.models import Event
from live_crew.core.timeslice import slice_index
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.interfaces.protocols import (
    ActionTransport,
    EventTransport,
    SchedulerBackend,
)
from live_crew.interfaces.orchestrator_protocol import OrchestratorProtocol
from live_crew.crewai_integration.wrapper import CrewAIWrapper
from live_crew.transports.console import ConsoleActionTransport


class MemoryScheduler(SchedulerBackend):
    """Memory-based scheduler focused on time slicing and event distribution.

    Handles time-based event processing and delegates complex orchestration
    to specialized orchestrator components. Maintains backward compatibility
    for simple single-crew scenarios.
    """

    def __init__(
        self,
        config: LiveCrewConfig,
        event_transport: EventTransport,
        action_transport: ActionTransport | None = None,
        context_backend: DictContextBackend | None = None,
        crew_registry: SimpleCrewRegistry | None = None,
        orchestrator: OrchestratorProtocol | None = None,
    ) -> None:
        """Initialize memory scheduler with optional orchestration delegation.

        Args:
            config: Live crew configuration
            event_transport: Transport for reading events
            action_transport: Transport for publishing actions (defaults to console)
            context_backend: Context storage backend (defaults to dict backend)
            crew_registry: Crew registry (defaults to empty registry)
            orchestrator: Orchestrator for multi-crew scenarios (optional)
        """
        self.config = config
        self.event_transport = event_transport
        self.action_transport = action_transport or ConsoleActionTransport()
        self.context_backend = context_backend or DictContextBackend()
        self.crew_registry = crew_registry or SimpleCrewRegistry()
        self.orchestrator = orchestrator  # Optional delegation target

        # Simple scheduler state (focused responsibility)
        self._epoch0: datetime | None = None
        self._processed_slices: set[int] = set()

    def set_epoch(self, epoch0: datetime) -> None:
        """Set the epoch start time for time slicing.

        Args:
            epoch0: The epoch start time (usually first event timestamp)
        """
        self._epoch0 = epoch0

    async def process_events(self) -> None:
        """Process all events from the event transport.

        Main entry point for the scheduler. Reads events, groups them by time slice,
        and processes each slice in chronological order to ensure deterministic behavior.
        """
        # Collect all events first
        events: list[Event[Any]] = []
        async for event in self.event_transport.subscribe_events():
            events.append(event)

        if not events:
            return

        # Set epoch to first event if not already set
        if self._epoch0 is None:
            self.set_epoch(events[0].ts)

        # Group events by time slice
        sliced_events: dict[int, list[Event[Any]]] = {}
        for event in events:
            slice_idx = slice_index(event.ts, self._epoch0, self.config.slice_ms)
            if slice_idx not in sliced_events:
                sliced_events[slice_idx] = []
            sliced_events[slice_idx].append(event)

        # Process slices in chronological order
        for slice_idx in sorted(sliced_events.keys()):
            await self._process_slice(slice_idx, sliced_events[slice_idx])

    async def _process_slice(self, slice_idx: int, events: list[Event[Any]]) -> None:
        """Process events in a specific time slice using delegation pattern.

        Args:
            slice_idx: The time slice index
            events: Events to process in this slice
        """
        if slice_idx in self._processed_slices:
            return  # Already processed

        if self.orchestrator and self._requires_orchestration():
            # Delegate to orchestrator for multi-crew scenarios
            await self.orchestrator.orchestrate_slice(
                slice_idx=slice_idx,
                events=events,
                context_backend=self.context_backend,
                action_transport=self.action_transport,
                crew_registry=self.crew_registry,
            )
        else:
            # Simple single-crew processing (maintains backward compatibility)
            for event in events:
                await self._process_event(event, slice_idx)

        self._processed_slices.add(slice_idx)

    def _requires_orchestration(self) -> bool:
        """Detect if complex orchestration is needed.

        Returns:
            True if multi-crew orchestration is required, False for simple processing
        """
        crew_wrappers = [
            handler
            for handler in [
                self.crew_registry.get_handler(crew_id)
                for crew_id in self.crew_registry.list_crews()
            ]
            if isinstance(handler, CrewAIWrapper)
        ]
        return len(crew_wrappers) > 1

    async def _process_event(self, event: Event[Any], slice_idx: int) -> None:
        """Process a single event with all applicable crews.

        Args:
            event: The event to process
            slice_idx: The time slice index
        """
        # Get context snapshot for this slice and stream
        context = await self.context_backend.get_snapshot(event.stream_id, slice_idx)

        # Process with all registered crews
        for crew_id in self.crew_registry.list_crews():
            handler = self.crew_registry.get_handler(crew_id)
            if handler is None:
                continue

            # Process event with crew handler
            try:
                actions = await handler.handle_event(event, context)

                # Publish all resulting actions
                for action in actions:
                    await self.action_transport.publish_action(action)

                    # Update context with action metadata
                    context_diff = {
                        f"last_action_{action.kind}": {
                            "ts": action.ts.isoformat(),
                            "crew_id": crew_id,
                            "payload": action.payload,
                        }
                    }
                    await self.context_backend.apply_diff(
                        event.stream_id, slice_idx, context_diff
                    )

            except Exception as e:
                # Basic error handling - log and continue processing
                # More sophisticated error handling can be added later
                print(f"Error processing event {event.kind} with crew {crew_id}: {e}")

    async def schedule_crew(
        self, crew_id: str, slice_idx: int, dependencies: list[Dependency]
    ) -> None:
        """Schedule a crew for execution in a specific slice.

        Args:
            crew_id: The crew identifier
            slice_idx: The slice index to schedule in
            dependencies: List of dependencies that must be satisfied

        Note:
            This implementation uses simple scheduling without complex dependency resolution.
            Enhanced dependency handling can be added for distributed scenarios.
        """
        # Simple implementation - just track that crew is scheduled
        # More complex dependency resolution can be added later
        pass

    async def mark_crew_complete(self, crew_id: str, slice_idx: int) -> None:
        """Mark a crew as completed for a specific slice.

        Args:
            crew_id: The crew identifier
            slice_idx: The slice index
        """
        # Simple tracking implementation
        pass

    async def get_pending_crews(self, slice_idx: int) -> list[str]:
        """Get list of crews pending execution for a slice.

        Args:
            slice_idx: The slice index to check

        Returns:
            List of crew IDs pending execution
        """
        # Return all registered crews as available for processing
        return self.crew_registry.list_crews()
