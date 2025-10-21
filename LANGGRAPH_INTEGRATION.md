# LangGraph Integration Summary

## Overview

This document summarizes the LangGraph integration added to live-crew, enabling users to run LangGraph workflows alongside CrewAI crews in the same event-driven orchestration system.

## What Was Added

### 1. Core Integration Module (`src/live_crew/langgraph_integration/`)

#### Models (`models.py`)
- **GraphRuntimeConfig**: Runtime configuration with Pydantic validation
  - Triggers, dependencies, timeout settings
  - LangGraph-specific: checkpointing, thread_id_strategy, interrupt points
- **GraphOrchestrationConfig**: Master configuration for multiple graphs
- **GraphConfig**: Individual graph configuration

#### Wrapper (`wrapper.py`)
- **LangGraphStateBridge**: Bidirectional state bridge
  - Tracks updates and deletions
  - Namespace isolation (global_ prefix for shared context)
  - Conversion between live-crew context and LangGraph state
- **LangGraphWrapper**: EventHandler protocol implementation
  - Converts Events to LangGraph initial state
  - Executes graph with checkpointing support
  - Converts final state to Actions (multiple patterns supported)
  - Thread ID generation for checkpoint persistence

#### Loader (`loader.py`)
- **LangGraphLoader**: Static methods for loading graphs
  - `load_yaml_graph()`: Load from directory with graph.py module
  - `load_python_graph()`: Load from direct Python definition
  - Factory function pattern support
  - Comprehensive error handling

### 2. Documentation

#### Reference Documentation (`.vibes/references/langgraph.md`)
- Comprehensive framework overview
- Core concepts (StateGraph, nodes, edges, checkpointing)
- Configuration approaches
- Integration patterns
- Best practices

#### README Updates
- Multi-framework support announcement
- Framework comparison table
- LangGraph usage patterns
- Integration examples

### 3. Example (`examples/langgraph_workflow/`)

Complete working example with:
- 3-node analysis workflow (analyze → validate → generate)
- Runtime configuration YAML
- Sample events
- Run script with orchestrator integration

### 4. Tests (`tests/test_langgraph_integration.py`)

Comprehensive test coverage:
- Configuration validation
- State bridge operations
- Event handling with multiple output patterns
- Thread ID generation
- Error handling

## Integration Architecture

### Event Flow

```
Event (live-crew)
    ↓
LangGraphWrapper.handle_event()
    ↓
LangGraphStateBridge.to_langgraph_state_format()
    ↓
langgraph_app.invoke(initial_state, config)
    ↓
LangGraphWrapper._convert_to_actions()
    ↓
Actions (live-crew)
```

### State Management

```
live-crew Context ←→ LangGraphStateBridge ←→ LangGraph State
                     
- Bidirectional updates
- Namespace isolation
- Type conversion
- Reserved keys handling
```

### Output Patterns Supported

1. **Actions List**: `state["actions"] = [{kind, payload}]`
2. **Messages List**: `state["messages"] = [message_objects]`
3. **Output Field**: `state["output"] = result`
4. **Full State**: All state fields converted to action payload

## Key Features

### 1. Checkpointing Support
- Thread-based persistence across events
- Configurable thread_id strategies:
  - `stream_id`: Per-stream persistence
  - `event_kind`: Per-event-type persistence
  - `custom`: User-defined

### 2. Human-in-the-Loop
- Interrupt before specified nodes
- Interrupt after specified nodes
- Resume from checkpoints

### 3. Context Bridge
- Read live-crew shared context in graph nodes
- Update context from graph execution
- Namespace isolation prevents conflicts
- Global context sharing with `global_` prefix

### 4. Flexible Configuration
- Python-defined graphs (primary)
- YAML configuration support (indirect)
- Runtime config in YAML
- Factory function pattern

## Usage Examples

### Basic Usage

```python
from live_crew import Orchestrator, LangGraphLoader

# Create and load graph
wrapper = LangGraphLoader.load_python_graph(
    graph_id="analysis",
    langgraph_app=create_graph(),
    runtime_config={
        "triggers": ["data_request"],
        "timeout_ms": 5000
    }
)

# Orchestrate
orchestrator = Orchestrator.from_file("events.json")
orchestrator.register_handler(wrapper, dependencies=[])
await orchestrator.run()
```

### With Checkpointing

```python
wrapper = LangGraphLoader.load_python_graph(
    graph_id="stateful_workflow",
    langgraph_app=create_graph_with_checkpointing(),
    runtime_config={
        "triggers": ["process_request"],
        "checkpointing": True,
        "thread_id_strategy": "stream_id",
        "timeout_ms": 10000
    }
)
```

### Multi-Framework Orchestration

```python
from live_crew import Orchestrator, CrewAILoader, LangGraphLoader

orchestrator = Orchestrator.from_file("events.json")

# CrewAI for autonomous analysis
crew = CrewAILoader.load_yaml_crew(
    crew_path="crews/analysis",
    runtime_config_path="crews/analysis/runtime.yaml"
)
orchestrator.register_handler(crew, dependencies=[])

# LangGraph for deterministic processing
graph = LangGraphLoader.load_python_graph(
    graph_id="pipeline",
    langgraph_app=my_graph,
    runtime_config={"triggers": ["analyzed_data"]}
)
orchestrator.register_handler(graph, dependencies=[("analysis_crew", -1)])

await orchestrator.run()
```

## Framework Comparison

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Philosophy** | Autonomous agents | Explicit control flow |
| **Best For** | Creative, exploratory | Deterministic, complex workflows |
| **Configuration** | YAML-first | Python-first |
| **State** | Memory patterns | Built-in state graph |
| **Control** | Agent decisions | Explicit routing |
| **Human-in-Loop** | Manual | Native breakpoints |
| **Integration** | Thin wrapper | Thin wrapper |

## Testing

### Test Coverage
- ✅ Configuration validation (Pydantic models)
- ✅ State bridge operations (CRUD)
- ✅ Event handling (all output patterns)
- ✅ Thread ID generation (all strategies)
- ✅ Error handling (wrapped exceptions)
- ✅ Loader functionality (Python and YAML paths)

### Running Tests

```bash
# Run LangGraph integration tests
pytest tests/test_langgraph_integration.py -v

# Run with coverage
pytest tests/test_langgraph_integration.py --cov=live_crew.langgraph_integration
```

## Dependencies Added

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
]
```

## Migration Path

For existing live-crew users:

1. **No Breaking Changes**: Existing CrewAI integrations work unchanged
2. **Opt-In**: LangGraph is optional, import only when needed
3. **Consistent API**: Same EventHandler protocol, same orchestration patterns
4. **Gradual Adoption**: Add LangGraph workflows one at a time

## Future Enhancements

Potential future improvements (not required for initial release):

1. **Advanced Checkpointing**: PostgreSQL/SQLite backends
2. **Streaming Support**: Handle LangGraph streaming execution
3. **Graph Visualization**: Debug and monitor graph execution
4. **Performance Metrics**: LangGraph-specific observability
5. **Complex Examples**: Multi-agent LangGraph workflows

## Security Considerations

- ✅ Input validation via Pydantic models
- ✅ Namespace isolation for state updates
- ✅ Exception wrapping for error containment
- ✅ No additional security vulnerabilities introduced
- ✅ Same security model as CrewAI integration

## Conclusion

LangGraph integration is **complete and production-ready**. Users can:
- Run LangGraph workflows within live-crew
- Combine CrewAI and LangGraph in the same application
- Leverage stateful execution and checkpointing
- Use human-in-the-loop workflows
- Maintain consistent event-driven architecture

The implementation follows live-crew's design principles:
- Minimal changes to existing code
- Protocol-based architecture
- Clean separation of concerns
- Comprehensive testing
- Production-ready quality

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- live-crew Documentation: README.md
- CrewAI Integration: src/live_crew/crewai_integration/
- Example Code: examples/langgraph_workflow/
