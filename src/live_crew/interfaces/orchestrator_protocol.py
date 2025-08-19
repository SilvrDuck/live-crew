"""Orchestrator protocol defining the interface for all orchestration implementations."""

from pathlib import Path
from typing import Any, Optional, Protocol, Union, List

from live_crew.config.settings import LiveCrewConfig
from live_crew.interfaces.protocols import EventHandler, ActionTransport
from live_crew.core.models import Event
from live_crew.backends.context import DictContextBackend
from live_crew.crew.registry import SimpleCrewRegistry


class OrchestrationResult:
    """Results from orchestration execution.

    This class is shared between all orchestrator implementations to provide
    consistent result reporting.
    """

    def __init__(
        self,
        events_processed: int,
        actions_generated: int,
        time_slices: int,
        actions: list[Any],
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
            f"events_processed={self.events_processed}, "
            f"actions_generated={self.actions_generated}, "
            f"time_slices={self.time_slices})"
        )


class OrchestratorProtocol(Protocol):
    """Protocol defining the interface for all orchestration implementations.

    This protocol ensures consistent APIs between vanilla live-crew orchestration
    and specialized implementations like CrewAI integration.
    """

    @property
    def config(self) -> LiveCrewConfig:
        """Get the live-crew configuration for this orchestrator."""
        ...

    @classmethod
    def from_file(
        cls, events_file: Union[str, Path], config: Optional[LiveCrewConfig] = None
    ) -> "OrchestratorProtocol":
        """Create orchestrator with file-based event transport.

        Args:
            events_file: Path to JSON file containing events
            config: Optional configuration object

        Returns:
            Configured orchestrator instance
        """
        ...

    @classmethod
    def from_config(cls, config_file: Union[str, Path]) -> "OrchestratorProtocol":
        """Create orchestrator from configuration file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Configured orchestrator instance
        """
        ...

    def register_handler(self, handler: EventHandler) -> None:
        """Register an event handler for processing events.

        Args:
            handler: The event handler to register
        """
        ...

    async def run(
        self, events_source: Optional[Union[str, Path]] = None
    ) -> OrchestrationResult:
        """Run orchestration and return results.

        Args:
            events_source: Optional path to events file. If not provided,
                          uses the source configured during initialization.

        Returns:
            Results from the orchestration execution
        """
        ...

    async def orchestrate_slice(
        self,
        slice_idx: int,
        events: List[Event[Any]],
        context_backend: DictContextBackend,
        action_transport: ActionTransport,
        crew_registry: SimpleCrewRegistry,
    ) -> None:
        """Orchestrate a single time slice with dependency-resolved crew execution.

        This method is used by schedulers for delegation of complex orchestration
        while maintaining separation of concerns between time slicing and crew coordination.

        Args:
            slice_idx: The time slice index
            events: Events to process in this slice
            context_backend: Context storage backend for state management
            action_transport: Transport for publishing resulting actions
            crew_registry: Registry containing crews to orchestrate
        """
        ...
