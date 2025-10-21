"""live-crew: Low-latency, slice-based orchestration for CrewAI crews and LangGraph workflows."""

# Primary API - covers 90% of use cases
from live_crew.core.models import Action, Event
from live_crew.config.settings import LiveCrewConfig, load_config, get_config
from live_crew.decorators import event_handler, HandlerBuilder
from live_crew.orchestration import Orchestrator, OrchestrationResult

# CrewAI Integration
from live_crew.crewai_integration import (
    CrewRuntimeConfig,
    CrewOrchestrationConfig,
    CrewConfig,
    CrewAILoader,
    CrewAIWrapper,
)

# LangGraph Integration
from live_crew.langgraph_integration import (
    GraphRuntimeConfig,
    GraphOrchestrationConfig,
    GraphConfig,
    LangGraphLoader,
    LangGraphWrapper,
)

# Secondary API - for customization needs
from live_crew.crew.definition import CrewDefinition
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.backends.context import DictContextBackend
from live_crew.core.timeslice import slice_index
from live_crew.transports.file import FileEventTransport
from live_crew.transports.console import ConsoleActionTransport
from live_crew.interfaces.results import CollectingResultCollector, NullResultCollector

# Advanced API - for power users who need protocol access
from live_crew.core.dependencies import CrewDep, EventDep, Dependency
from live_crew.interfaces import protocols

__all__ = [
    # === PRIMARY API (covers 90% of use cases) ===
    # Core orchestration
    "Orchestrator",
    "OrchestrationResult",
    # Event handling
    "event_handler",
    # Data models with convenience methods
    "Event",
    "Action",
    # Configuration essentials
    "load_config",
    "get_config",
    # === CREWAI INTEGRATION ===
    "CrewRuntimeConfig",
    "CrewOrchestrationConfig",
    "CrewConfig",
    "CrewAILoader",
    "CrewAIWrapper",
    # === LANGGRAPH INTEGRATION ===
    "GraphRuntimeConfig",
    "GraphOrchestrationConfig",
    "GraphConfig",
    "LangGraphLoader",
    "LangGraphWrapper",
    # === SECONDARY API (for customization) ===
    # Crew management (declarative approach)
    "CrewDefinition",
    "SimpleCrewRegistry",
    # Configuration objects
    "LiveCrewConfig",
    # Context & timing utilities
    "DictContextBackend",
    "slice_index",
    # Common transports
    "FileEventTransport",
    "ConsoleActionTransport",
    # Result collection
    "CollectingResultCollector",
    "NullResultCollector",
    # === ADVANCED API (for power users) ===
    # Dependencies (declarative orchestration)
    "CrewDep",
    "EventDep",
    "Dependency",
    # Programmatic handler creation
    "HandlerBuilder",
    # Protocol access for custom implementations
    "protocols",
]
