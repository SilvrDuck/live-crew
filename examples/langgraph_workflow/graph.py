"""Simple LangGraph workflow for data analysis.

This module defines a LangGraph workflow that processes data analysis requests
through multiple stages. It demonstrates integration with live-crew's event-driven
orchestration system.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END


class AnalysisState(TypedDict):
    """State schema for the analysis workflow."""

    input: str
    analysis: str
    validation_result: str
    output: str
    actions: list


def analyze_node(state: AnalysisState) -> AnalysisState:
    """Analyze the input data.

    Args:
        state: Current workflow state

    Returns:
        Updated state with analysis results
    """
    input_data = state.get("input", "")
    state["analysis"] = f"Analyzed: {input_data}"
    state["actions"] = state.get("actions", [])
    return state


def validate_node(state: AnalysisState) -> AnalysisState:
    """Validate the analysis results.

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation results
    """
    analysis = state.get("analysis", "")
    state["validation_result"] = f"Valid analysis: {len(analysis)} characters"
    return state


def generate_node(state: AnalysisState) -> AnalysisState:
    """Generate final output based on validated analysis.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final output and actions
    """
    analysis = state.get("analysis", "")
    validation = state.get("validation_result", "")

    state["output"] = f"Report: {analysis} | {validation}"

    # Add action for live-crew to process
    state["actions"] = [
        {
            "kind": "analysis_complete",
            "payload": {"result": state["output"], "status": "success"},
        }
    ]

    return state


def create_graph():
    """Factory function to create and compile the analysis workflow.

    Returns:
        Compiled LangGraph application ready for execution
    """
    # Create the graph with state schema
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("generate", generate_node)

    # Define edges (workflow flow)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", END)

    # Compile and return
    return workflow.compile()


# Alternative: Create graph with checkpointing
def create_graph_with_checkpointing():
    """Factory function to create graph with checkpointing enabled.

    Returns:
        Compiled LangGraph application with checkpointing
    """
    from langgraph.checkpoint.memory import MemorySaver

    workflow = StateGraph(AnalysisState)

    # Add nodes (same as above)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("generate", generate_node)

    # Define edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", END)

    # Compile with checkpointing
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
