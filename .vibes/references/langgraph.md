# LangGraph Framework Reference

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. LangGraph is designed for creating agent and multi-agent workflows with built-in persistence, human-in-the-loop, and streaming support.

## Core Concepts

### 1. StateGraph
The fundamental building block for creating stateful workflows:
- **Nodes**: Functions that process state
- **Edges**: Define the flow between nodes
- **State**: Typed dictionary that persists across nodes
- **Checkpointing**: Automatic state persistence at each step

### 2. Graph Components

#### Nodes
```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: list[str]
    next_action: str

def process_node(state: AgentState) -> AgentState:
    """Node function that processes and returns updated state."""
    state["messages"].append("Processed")
    return state

graph = StateGraph(AgentState)
graph.add_node("processor", process_node)
```

#### Edges
```python
# Normal edges (unconditional)
graph.add_edge("node_a", "node_b")

# Conditional edges (dynamic routing)
def route_decision(state: AgentState) -> str:
    if state["next_action"] == "continue":
        return "node_b"
    return "node_c"

graph.add_conditional_edges(
    "node_a",
    route_decision,
    {
        "node_b": "node_b",
        "node_c": "node_c"
    }
)
```

#### Entry and End Points
```python
# Set entry point
graph.set_entry_point("start_node")

# Set finish points
graph.set_finish_point("end_node")

# Compile the graph
app = graph.compile()
```

### 3. State Management

#### State Schema
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # Messages with reducer function
    messages: Annotated[list[str], add_messages]
    # Simple fields
    user_id: str
    context: dict
    # Counter with custom reducer
    step_count: int
```

#### Built-in Reducers
- `add_messages`: Appends messages intelligently (deduplication)
- Custom reducers can be defined for any field

### 4. Checkpointing and Persistence

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpointing
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Execute with thread_id for persistence
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config=config)

# Resume from checkpoint
continued = app.invoke(None, config=config)  # Continues from last checkpoint
```

### 5. Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

def human_approval_node(state: AgentState) -> AgentState:
    """Node that requires human approval."""
    return state

graph.add_node("human_approval", human_approval_node)

# Create breakpoint for human intervention
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_approval"]  # Pause before this node
)

# Execute until breakpoint
app.invoke(initial_state, config={"thread_id": "123"})

# Human reviews, then continues
app.invoke(None, config={"thread_id": "123"})  # Resumes execution
```

## Configuration Approaches

### 1. Pure Python Definition (Primary)

LangGraph is primarily code-first, with graphs defined programmatically:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class WorkflowState(TypedDict):
    input: str
    output: str
    step: int

# Define nodes
def analyze_input(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Analyzed: {state['input']}"
    state["step"] = 1
    return state

def generate_response(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Response: {state['output']}"
    state["step"] = 2
    return state

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

### 2. YAML Configuration (Indirect)

LangGraph doesn't have native YAML support, but can be configured through YAML:

```yaml
# workflow_config.yaml
workflow:
  name: "analysis_workflow"
  state_schema:
    input: str
    output: str
    step: int
  
  nodes:
    - name: "analyze"
      function: "analysis.analyze_input"
    - name: "generate"
      function: "generation.generate_response"
  
  edges:
    - from: "analyze"
      to: "generate"
  
  entry_point: "analyze"
  end_points:
    - "generate"
```

**Loading from YAML (custom implementation)**:
```python
import yaml
from pathlib import Path
from typing import Any, Callable
import importlib

def load_function(function_path: str) -> Callable:
    """Load a function from module.function_name string."""
    module_name, func_name = function_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

def load_graph_from_yaml(config_path: Path) -> StateGraph:
    """Load LangGraph from YAML configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    workflow_config = config["workflow"]
    
    # Create StateGraph (state schema needs to be defined in Python)
    from workflows.states import get_state_schema
    state_schema = get_state_schema(workflow_config["state_schema"])
    graph = StateGraph(state_schema)
    
    # Add nodes
    for node_config in workflow_config["nodes"]:
        func = load_function(node_config["function"])
        graph.add_node(node_config["name"], func)
    
    # Add edges
    for edge_config in workflow_config["edges"]:
        graph.add_edge(edge_config["from"], edge_config["to"])
    
    # Set entry and end points
    graph.set_entry_point(workflow_config["entry_point"])
    for end_point in workflow_config.get("end_points", []):
        graph.add_edge(end_point, END)
    
    return graph.compile()
```

## Key Differences from CrewAI

| Aspect | LangGraph | CrewAI |
|--------|-----------|---------|
| **Philosophy** | Explicit control flow, stateful | Autonomous agents, role-based |
| **State Management** | Built-in state graph with reducers | Memory/context patterns |
| **Configuration** | Code-first (Python) | YAML-first (agents/tasks) |
| **Execution Model** | Graph traversal with nodes/edges | Sequential/hierarchical crews |
| **Control** | Deterministic routing | Autonomous decision-making |
| **Human-in-Loop** | Native breakpoints | Manual implementation |
| **Persistence** | Built-in checkpointing | External memory systems |
| **Tools** | LangChain tools integration | CrewAI tools system |

## Integration Patterns for live-crew

### Pattern 1: Graph as Event Handler

```python
from langgraph.graph import StateGraph, END
from live_crew.core.models import Event, Action

class EventWorkflowState(TypedDict):
    event: Event
    context: dict
    actions: list[Action]

def create_event_graph() -> StateGraph:
    """Create LangGraph that processes live-crew events."""
    graph = StateGraph(EventWorkflowState)
    
    def process_event(state: EventWorkflowState) -> EventWorkflowState:
        event = state["event"]
        # Process event logic
        action = Action(
            ts=event.ts,
            kind="processed",
            stream_id=event.stream_id,
            payload={"result": "processed"}
        )
        state["actions"].append(action)
        return state
    
    graph.add_node("process", process_event)
    graph.set_entry_point("process")
    graph.add_edge("process", END)
    
    return graph.compile()
```

### Pattern 2: Stateful Multi-Step Workflows

```python
from langgraph.checkpoint.memory import MemorySaver

def create_stateful_workflow() -> StateGraph:
    """Create stateful workflow with persistence."""
    graph = StateGraph(WorkflowState)
    
    # Multi-step processing
    graph.add_node("step1", step1_processor)
    graph.add_node("step2", step2_processor)
    graph.add_node("step3", step3_processor)
    
    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)
    
    # Compile with checkpointing
    return graph.compile(checkpointer=MemorySaver())
```

### Pattern 3: Conditional Routing

```python
def create_routing_workflow() -> StateGraph:
    """Create workflow with conditional routing."""
    graph = StateGraph(RoutingState)
    
    def route_condition(state: RoutingState) -> str:
        if state["score"] > 0.8:
            return "high_priority"
        return "normal_priority"
    
    graph.add_node("classifier", classifier_node)
    graph.add_node("high_priority", high_priority_handler)
    graph.add_node("normal_priority", normal_priority_handler)
    
    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_condition,
        {
            "high_priority": "high_priority",
            "normal_priority": "normal_priority"
        }
    )
    
    graph.add_edge("high_priority", END)
    graph.add_edge("normal_priority", END)
    
    return graph.compile()
```

## Best Practices for live-crew Integration

### 1. State Design
- Keep state schema simple and focused
- Use reducers for list/accumulator fields
- Include event metadata in state for traceability

### 2. Node Design
- Each node should be a pure function of state
- Nodes should be small and focused on single responsibility
- Use type hints for all node functions

### 3. Context Integration
- Bridge LangGraph state with live-crew context
- Map event payload to graph initial state
- Convert graph final state to actions

### 4. Checkpointing Strategy
- Use MemorySaver for development
- Consider distributed checkpointing for production
- Thread IDs should map to stream_ids for consistency

### 5. Error Handling
- Wrap node execution in try-catch
- Use error nodes for recovery paths
- Propagate errors to live-crew orchestration layer

### 6. Performance
- Keep graph shallow (fewer hops)
- Avoid heavy computation in routing functions
- Consider async node execution where possible

## LangGraph vs LangChain

**LangChain**: Linear chains, simpler for straightforward workflows
**LangGraph**: Cyclic graphs, better for complex multi-step workflows with branching

Choose LangGraph when you need:
- Loops and cycles in workflow
- Conditional branching based on state
- Human-in-the-loop checkpoints
- State persistence across runs
- Complex multi-actor coordination

## Resources

- **Documentation**: https://langchain-ai.github.io/langgraph/
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Tutorials**: LangGraph how-to guides
- **Community**: LangChain Discord #langgraph channel

## Installation

```bash
# Core LangGraph
uv add langgraph

# With checkpointing support
uv add langgraph[memory]

# Full installation
uv add "langgraph[all]"
```

## Version Compatibility

- **Python**: 3.9+ (live-crew requires 3.10+)
- **LangChain**: Compatible with LangChain 0.1+
- **Pydantic**: v2 (same as live-crew)
