"""LangGraph integration for live-crew orchestration.

This package provides support for integrating LangGraph workflows with live-crew's
event-driven orchestration system. It mirrors the CrewAI integration architecture
while adapting to LangGraph's graph-based execution model.

Key Components:
    - models: Pydantic configuration models for LangGraph workflows
    - loader: Load LangGraph graphs from Python definitions and YAML config
    - wrapper: Adapter implementing EventHandler protocol for graph execution
    - state_bridge: Bidirectional bridge between LangGraph state and live-crew context

Example Usage:
    ```python
    from live_crew.langgraph_integration import LangGraphLoader, LangGraphWrapper
    
    # Load graph from Python definition
    wrapper = LangGraphLoader.load_python_graph(
        graph_id="analysis",
        langgraph_app=my_compiled_graph,
        runtime_config={
            "triggers": ["data_received"],
            "timeout_ms": 5000
        }
    )
    
    # Use in orchestrator
    orchestrator.register_handler(wrapper, dependencies=[])
    ```
"""

from live_crew.langgraph_integration.models import (
    GraphRuntimeConfig,
    GraphOrchestrationConfig,
    GraphConfig,
)
from live_crew.langgraph_integration.loader import (
    LangGraphLoader,
    LangGraphLoadError,
    LangGraphConfigError,
)
from live_crew.langgraph_integration.wrapper import (
    LangGraphWrapper,
    LangGraphExecutionError,
    LangGraphStateBridge,
)

__all__ = [
    # Configuration models
    "GraphRuntimeConfig",
    "GraphOrchestrationConfig",
    "GraphConfig",
    # Loader
    "LangGraphLoader",
    "LangGraphLoadError",
    "LangGraphConfigError",
    # Wrapper and bridge
    "LangGraphWrapper",
    "LangGraphExecutionError",
    "LangGraphStateBridge",
]
