"""CrewAI orchestrator implementing OrchestratorProtocol for CrewAI integration."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import ValidationError

from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import LiveCrewConfig, get_config, load_config
from live_crew.core.models import Event
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.crewai_integration.loader import CrewAILoader, CrewAIConfigError
from live_crew.crewai_integration.models import CrewOrchestrationConfig
from live_crew.crewai_integration.wrapper import CrewAIWrapper, CrewAIExecutionError
from live_crew.interfaces.orchestrator_protocol import (
    OrchestratorProtocol,
    OrchestrationResult,
)
from live_crew.interfaces.protocols import EventHandler, ActionTransport
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.console import ConsoleActionTransport
from live_crew.transports.file import FileEventTransport


class CrewOrchestrator(OrchestratorProtocol):
    """CrewAI orchestrator implementing OrchestratorProtocol.

    This class provides direct orchestration for CrewAI crews, implementing the same
    protocol as the main Orchestrator but specialized for CrewAI integration.

    Example YAML approach:
        orchestrator = CrewOrchestrator.from_config("live_crew_config.yaml")
        await orchestrator.run()

    Example Python approach:
        orchestrator = CrewOrchestrator()
        orchestrator.register_crew("analytics", my_crew, triggers=["user_signup"])
        await orchestrator.run("events.json")
    """

    def __init__(
        self,
        config: Optional[LiveCrewConfig] = None,
        event_transport: Optional[FileEventTransport] = None,
    ):
        """Initialize the CrewAI orchestrator.

        Args:
            config: Optional live-crew configuration. If not provided, uses get_config().
            event_transport: Optional event transport. If not provided, must use from_file().
        """
        self._config = config or get_config()
        self._crew_wrappers: List[CrewAIWrapper] = []

        # Initialize components immediately (no late binding)
        self._event_transport = event_transport
        self._action_transport = ConsoleActionTransport()
        self._context_backend = DictContextBackend()
        self._crew_registry = SimpleCrewRegistry()

    @property
    def config(self) -> LiveCrewConfig:
        """Get the live-crew configuration for this orchestrator."""
        return self._config

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "CrewOrchestrator":
        """Create CrewOrchestrator from YAML configuration file.

        Args:
            config_path: Path to the orchestration configuration YAML file

        Returns:
            Configured CrewOrchestrator instance ready for execution
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise CrewAIConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            orchestration_config = CrewOrchestrationConfig(**raw_config)
        except yaml.YAMLError as e:
            raise CrewAIConfigError(f"Invalid YAML in config file: {e}") from e
        except ValidationError as e:
            raise CrewAIConfigError(f"Invalid orchestration configuration: {e}") from e

        # Create config with optional overrides
        live_crew_config = load_config() if orchestration_config.slice_ms else None
        if live_crew_config and orchestration_config.slice_ms:
            live_crew_config.slice_ms = orchestration_config.slice_ms

        orchestrator = cls(config=live_crew_config)

        # Load all crews from configuration
        for crew_config in orchestration_config.crews:
            crew_path = Path(crew_config.path)
            if not crew_path.is_absolute():
                crew_path = config_path.parent / crew_path

            runtime_path = Path(crew_config.runtime)
            if not runtime_path.is_absolute():
                runtime_path = crew_path / runtime_path

            wrapper = CrewAILoader.load_yaml_crew(crew_path, runtime_path)
            orchestrator._crew_wrappers.append(wrapper)

        return orchestrator

    @classmethod
    def from_file(
        cls, events_file: Union[str, Path], config: Optional[LiveCrewConfig] = None
    ) -> "CrewOrchestrator":
        """Create CrewOrchestrator with file-based event transport.

        Args:
            events_file: Path to JSON file containing events
            config: Optional configuration object

        Returns:
            Configured orchestrator instance
        """
        events_path = Path(events_file)
        file_transport = FileEventTransport(events_path)
        return cls(config=config, event_transport=file_transport)

    def register_crew(self, crew_id: str, crewai_crew: Any, **kwargs) -> None:
        """Register a CrewAI crew using Python approach.

        Args:
            crew_id: Unique identifier for this crew
            crewai_crew: CrewAI Crew instance (agents, tasks, process configured)
            **kwargs: Runtime configuration (triggers, timeout_ms, etc.)
        """
        wrapper = CrewAILoader.load_python_crew(crew_id, crewai_crew, kwargs)
        self._crew_wrappers.append(wrapper)

    def register_handler(self, handler: EventHandler) -> None:
        """Register an event handler for processing events.

        Args:
            handler: The event handler to register
        """
        # For CrewOrchestrator, we expect CrewAI wrappers, but allow generic handlers too
        if hasattr(handler, "crew_id") and hasattr(handler, "handle_event"):
            self._crew_wrappers.append(handler)  # type: ignore
        else:
            # Register with crew registry for compatibility
            self._crew_registry.register_handler(handler, [])

    async def run(
        self, events_source: Optional[Union[str, Path]] = None
    ) -> OrchestrationResult:
        """Run orchestration with direct CrewAI-specific logic.

        Args:
            events_source: Optional path to events file

        Returns:
            Results from the orchestration execution
        """
        if not self._crew_wrappers and not self._crew_registry._handlers:
            raise CrewAIConfigError(
                "No crews registered. Use from_config() or register_crew() first."
            )

        # Determine event transport
        event_transport = self._event_transport
        if events_source:
            event_transport = FileEventTransport(Path(events_source))
        elif not event_transport:
            raise CrewAIConfigError(
                "No events source provided. Use run(events_file) or from_file()."
            )

        try:
            # Create scheduler with CrewAI-specific configuration
            scheduler = MemoryScheduler(
                self._config,
                event_transport,
                self._action_transport,
                self._context_backend,
            )

            # Register all CrewAI wrappers with the scheduler
            for wrapper in self._crew_wrappers:
                # Register wrapper as crew
                scheduler.crew_registry.register_crew(wrapper, [])

            # Register any additional handlers
            for handler_id, handler in self._crew_registry._handlers.items():
                scheduler.crew_registry.register_crew(handler, [])

            # Process events through the scheduler
            await scheduler.process_events()

            # Create orchestration result
            # Note: In a full implementation, we'd collect actual metrics from the scheduler
            return OrchestrationResult(
                events_processed=0,  # Would be populated by scheduler
                actions_generated=0,  # Would be populated by scheduler
                time_slices=0,  # Would be populated by scheduler
                actions=[],  # Would be populated by action transport
                context_final_state={},  # Would be populated by context backend
            )

        except Exception as e:
            raise CrewAIExecutionError(f"CrewAI orchestration failed: {str(e)}") from e

    def list_crews(self) -> List[Dict[str, Any]]:
        """List all registered CrewAI crews with their configuration.

        Returns:
            List of crew information dictionaries
        """
        return [
            {
                "crew_id": wrapper.crew_id,
                "triggers": wrapper.triggers,
                "timeout_ms": wrapper.timeout_ms,
                "crew_type": type(wrapper.crewai_crew).__name__,
            }
            for wrapper in self._crew_wrappers
        ]

    def get_crew_info(self, crew_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific crew.

        Args:
            crew_id: The crew identifier to look up

        Returns:
            Crew information dictionary or None if not found
        """
        for wrapper in self._crew_wrappers:
            if wrapper.crew_id == wrapper.crew_id:
                return {
                    "crew_id": wrapper.crew_id,
                    "triggers": wrapper.triggers,
                    "timeout_ms": wrapper.timeout_ms,
                    "crew_type": type(wrapper.crewai_crew).__name__,
                    "crewai_crew": wrapper.crewai_crew,
                }
        return None

    async def orchestrate_slice(
        self,
        slice_idx: int,
        events: List[Event[Any]],
        context_backend: DictContextBackend,
        action_transport: ActionTransport,
        crew_registry: SimpleCrewRegistry,
    ) -> None:
        """Orchestrate a single time slice with CrewAI multi-crew coordination.

        This method implements the complex orchestration logic with dependency resolution,
        execution tracking, and failure handling that the MemoryScheduler delegates to us.

        Args:
            slice_idx: The time slice index
            events: Events to process in this slice
            context_backend: Context storage backend for state management
            action_transport: Transport for publishing resulting actions
            crew_registry: Registry containing crews to orchestrate
        """
        from live_crew.crewai_integration.dependency_resolver import (
            TopologicalDependencyResolver,
        )
        from live_crew.crewai_integration.execution_tracker import ExecutionTracker
        from live_crew.crewai_integration.failure_handler import PartialFailureHandler

        # Get CrewAI wrappers to orchestrate
        crew_wrappers = [
            handler
            for handler in [
                crew_registry.get_handler(crew_id)
                for crew_id in crew_registry.list_crews()
            ]
            if isinstance(handler, CrewAIWrapper)
        ] + self._crew_wrappers

        if not crew_wrappers:
            return  # No crews to orchestrate

        try:
            # Initialize orchestration components
            dependency_resolver = TopologicalDependencyResolver()
            execution_tracker = ExecutionTracker()
            failure_handler = PartialFailureHandler()

            # Resolve execution phases with dependencies
            execution_phases = dependency_resolver.resolve_execution_order(
                crew_wrappers, slice_idx
            )

            # Create execution context for tracking
            execution_context = execution_tracker.create_execution_context(
                slice_idx, execution_phases
            )

            # Execute crews in dependency-resolved phases
            for phase in execution_phases:
                await self._execute_crew_phase(
                    phase.crews,
                    events,
                    execution_context,
                    context_backend,
                    action_transport,
                    failure_handler,
                )

            # Finalize execution context
            execution_tracker.finalize_execution(execution_context)

        except Exception as e:
            # Log orchestration failure but don't crash the scheduler
            print(f"Multi-crew orchestration failed for slice {slice_idx}: {e}")

    async def _execute_crew_phase(
        self,
        crew_ids: List[str],
        events: List[Event[Any]],
        execution_context,
        context_backend: DictContextBackend,
        action_transport: ActionTransport,
        failure_handler,
    ) -> None:
        """Execute a phase of crews concurrently with proper error handling."""
        import asyncio

        # Create tasks for concurrent execution
        tasks = []
        for crew_id in crew_ids:
            # Find the crew wrapper
            crew_wrapper = None
            for wrapper in self._crew_wrappers:
                if wrapper.crew_id == crew_id:
                    crew_wrapper = wrapper
                    break

            if crew_wrapper:
                task = asyncio.create_task(
                    self._execute_single_crew(
                        crew_wrapper,
                        events,
                        execution_context,
                        context_backend,
                        action_transport,
                    )
                )
                tasks.append((crew_id, task))

        # Wait for all crews in this phase to complete
        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            # Process results and handle failures
            for (crew_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    dependent_crews = []  # Would need dependency graph to calculate
                    await failure_handler.handle_crew_failure(
                        crew_id, result, execution_context, dependent_crews
                    )

    async def _execute_single_crew(
        self,
        crew_wrapper: CrewAIWrapper,
        events: List[Event[Any]],
        execution_context,
        context_backend: DictContextBackend,
        action_transport: ActionTransport,
    ) -> None:
        """Execute single crew with comprehensive tracking."""
        from live_crew.crewai_integration.execution_tracker import ExecutionTracker

        execution_tracker = ExecutionTracker()
        execution_tracker.mark_crew_started(execution_context, crew_wrapper.crew_id)

        try:
            all_actions = []
            context_updates = {}

            # Process all events with this crew
            for event in events:
                # Get context snapshot
                shared_context = await context_backend.get_snapshot(
                    event.stream_id, execution_context.slice_idx
                )

                # Execute crew handler
                actions = await crew_wrapper.handle_event(event, shared_context)
                all_actions.extend(actions)

                # Publish actions
                for action in actions:
                    await action_transport.publish_action(action)

                # Apply context updates
                await context_backend.apply_diff(
                    event.stream_id, execution_context.slice_idx, shared_context
                )

            # Mark crew as completed
            execution_tracker.mark_crew_completed(
                execution_context, crew_wrapper.crew_id, all_actions, context_updates
            )

        except Exception as e:
            # Let the failure propagate to be handled by the phase executor
            raise e
