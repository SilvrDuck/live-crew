"""High-level orchestration API for live-crew framework.

This module provides a simplified facade over the protocol-based architecture,
hiding complexity while maintaining full extensibility for advanced users.
"""

from pathlib import Path
from typing import Any, Optional, Union

from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import LiveCrewConfig, get_config, load_config
from live_crew.core.models import Action
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.interfaces.protocols import (
    ActionTransport,
    ContextBackend,
    CrewRegistry,
    EventHandler,
    EventTransport,
)
from live_crew.interfaces.results import (
    ResultCollector,
    CollectingResultCollector,
    NullResultCollector,
)
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.console import ConsoleActionTransport
from live_crew.transports.file import FileEventTransport


class OrchestrationResult:
    """Results from orchestration execution."""

    def __init__(
        self,
        events_processed: int,
        actions_generated: int,
        time_slices: int,
        actions: list[Action[Any]],
        context_final_state: dict[str, Any],
    ):
        self.events_processed = events_processed
        self.actions_generated = actions_generated
        self.time_slices = time_slices
        self.actions = actions
        self.context_final_state = context_final_state

    def __repr__(self) -> str:
        return (
            f"OrchestrationResult("
            f"events={self.events_processed}, "
            f"actions={self.actions_generated}, "
            f"slices={self.time_slices})"
        )


class Orchestrator:
    """High-level orchestration API that simplifies live-crew usage.

    This class provides a facade over the protocol-based architecture,
    offering sensible defaults while maintaining full extensibility.

    Examples:
        Simple file-based processing:
        >>> orchestrator = Orchestrator.from_file("events.json")
        >>> result = await orchestrator.run()

        Configuration-driven setup:
        >>> orchestrator = Orchestrator.from_config("config.yaml")
        >>> result = await orchestrator.run()

        Advanced customization:
        >>> orchestrator = Orchestrator(
        ...     config=custom_config,
        ...     event_transport=custom_transport,
        ... )
    """

    def __init__(
        self,
        config: Optional[LiveCrewConfig] = None,
        event_transport: Optional[EventTransport] = None,
        action_transport: Optional[ActionTransport] = None,
        context_backend: Optional[ContextBackend] = None,
        crew_registry: Optional[CrewRegistry] = None,
        result_collector: Optional[ResultCollector] = None,
    ):
        """Initialize orchestrator with optional custom components.

        Args:
            config: Configuration object. Uses defaults if None.
            event_transport: Custom event transport. Uses FileEventTransport if None.
            action_transport: Custom action transport. Uses ConsoleActionTransport if None.
            context_backend: Custom context backend. Uses DictContextBackend if None.
            crew_registry: Custom crew registry. Uses SimpleCrewRegistry if None.
            result_collector: Custom result collector. Auto-selected based on transport if None.
        """
        self._config = config or get_config()
        self._event_transport = event_transport
        self._action_transport = action_transport or ConsoleActionTransport()
        self._context_backend = context_backend or DictContextBackend()
        self._crew_registry = crew_registry or SimpleCrewRegistry()

        # Use provided collector or create appropriate default
        if result_collector is None:
            result_collector = self._create_default_collector()
        self._result_collector = result_collector

        # Will be set when scheduler is created
        self._scheduler: Optional[MemoryScheduler] = None

    @classmethod
    def from_file(
        cls,
        events_file: Union[str, Path],
        config: Optional[LiveCrewConfig] = None,
    ) -> "Orchestrator":
        """Create orchestrator for processing events from a file.

        Args:
            events_file: Path to JSON file containing events
            config: Optional configuration object

        Returns:
            Configured orchestrator instance

        Raises:
            PathSecurityError: If file path fails security validation
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        events_path = Path(events_file)
        file_transport = FileEventTransport(events_path)

        return cls(
            config=config,
            event_transport=file_transport,
        )

    @classmethod
    def from_config(cls, config_file: Union[str, Path]) -> "Orchestrator":
        """Create orchestrator from configuration file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Configured orchestrator instance

        Note:
            This currently loads the config but uses default transports.
            Enhanced configuration support will be added in Phase 2.
        """
        # Load configuration from specified file
        config = load_config(config_file)

        return cls(config=config)

    def register_handler(
        self,
        handler: EventHandler,
        event_types: Optional[list[str]] = None,
    ) -> None:
        """Register an event handler with optional event type filtering.

        Args:
            handler: Handler implementing EventHandler protocol
            event_types: Optional list of event types to handle.
                        If None, handler receives all events.
        """
        self._crew_registry.register_crew(handler, event_types or [])

    def _create_default_collector(self) -> ResultCollector:
        """Factory method for default result collection strategy.

        Returns:
            Appropriate ResultCollector based on action transport capabilities
        """
        # Check if action transport supports collection
        if hasattr(self._action_transport, "actions"):
            return CollectingResultCollector(self._action_transport)
        else:
            # For streaming/console transports, use NullResultCollector
            return NullResultCollector()

    def _ensure_scheduler(self) -> MemoryScheduler:
        """Ensure scheduler is created with current configuration."""
        if self._scheduler is None:
            if self._event_transport is None:
                raise ValueError(
                    "No event transport configured. Use from_file() or provide event_transport."
                )

            self._scheduler = MemoryScheduler(
                config=self._config,
                event_transport=self._event_transport,
                action_transport=self._action_transport,
                context_backend=self._context_backend,
                crew_registry=self._crew_registry,
            )

        return self._scheduler

    async def run(self) -> OrchestrationResult:
        """Run orchestration and return results.

        Returns:
            OrchestrationResult with processing statistics and outputs

        Raises:
            ValueError: If no event transport is configured
        """
        scheduler = self._ensure_scheduler()

        # Process all events
        await scheduler.process_events()

        # Collect results using result collector strategy
        events = await self._event_transport.read_events()
        actions = await self._result_collector.get_actions()

        # Update collector with basic statistics
        self._result_collector.update_statistics(
            {
                "events_processed": len(events),
                "time_slices": len(scheduler._processed_slices),
            }
        )

        # Get comprehensive statistics from collector
        stats = await self._result_collector.get_statistics()

        return OrchestrationResult(
            events_processed=stats.get("events_processed", len(events)),
            actions_generated=len(actions),
            time_slices=stats.get("time_slices", 0),
            actions=actions,
            context_final_state=self._context_backend.get_all_contexts(),
        )

    @property
    def config(self) -> LiveCrewConfig:
        """Get the current configuration."""
        return self._config

    @property
    def crew_registry(self) -> CrewRegistry:
        """Get the crew registry for advanced operations."""
        return self._crew_registry
