# LangGraph Workflow Example

This example demonstrates how to integrate a LangGraph workflow with live-crew's event-driven orchestration system.

## Overview

This example shows:
1. A simple LangGraph workflow with multiple nodes
2. Integration with live-crew's event system
3. State management and context bridging
4. Conversion of graph outputs to live-crew actions

## Workflow

The example workflow processes data analysis requests through multiple stages:
1. **Analyze Node**: Receives input and performs initial analysis
2. **Validate Node**: Validates the analysis results
3. **Generate Node**: Generates final output based on validated analysis

## Files

- `graph.py`: LangGraph workflow definition
- `simple_workflow.runtime.yaml`: live-crew orchestration configuration
- `run_example.py`: Script to run the example
- `events.json`: Sample events for testing

## Running the Example

```bash
# Install dependencies
uv sync --dev

# Run the example
python examples/langgraph_workflow/run_example.py
```

## Expected Output

The workflow will:
1. Process "data_request" events
2. Execute the LangGraph workflow
3. Generate actions with analysis results
4. Display output to console

## Integration Pattern

This example follows the **Python-defined graph** pattern:
- Graph is defined directly in Python (graph.py)
- Runtime configuration in YAML (simple_workflow.runtime.yaml)
- Minimal glue code in run_example.py

The same workflow can be integrated using the **YAML-configured** pattern by organizing files according to the loader's expectations.
