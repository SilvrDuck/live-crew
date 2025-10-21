"""Pydantic models for LangGraph integration configuration."""

from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict

from live_crew.core.dependencies import Dependency


class GraphRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Runtime configuration for LangGraph workflow integration with live-crew orchestration.

    This model defines how a LangGraph workflow integrates with live-crew's
    event-driven orchestration system, including triggers, dependencies, and
    execution parameters.
    """

    graph: str = Field(
        min_length=1,
        description="Unique identifier for this graph within live-crew orchestration",
    )

    triggers: List[str] = Field(
        min_length=1,
        description="List of event kinds that trigger this graph's execution",
    )

    needs: Optional[List[Dependency]] = Field(
        default=None,
        description="Dependencies that must be satisfied before graph execution",
    )

    wait_policy: Literal["any", "all", "none"] = Field(
        default="none", description="Policy for handling dependency satisfaction"
    )

    timeout_ms: int = Field(
        default=5000,
        gt=0,
        le=300000,  # Max 5 minutes
        description="Maximum execution time for graph processing in milliseconds",
    )

    slice_stride: int = Field(
        default=1, ge=1, description="Execute graph every N slices (1 = every slice)"
    )

    # LangGraph-specific configuration
    checkpointing: bool = Field(
        default=False,
        description="Enable LangGraph checkpointing for stateful execution across slices",
    )

    thread_id_strategy: Literal["stream_id", "event_kind", "custom"] = Field(
        default="stream_id",
        description="Strategy for generating LangGraph thread IDs for checkpoint persistence",
    )

    interrupt_before: Optional[List[str]] = Field(
        default=None,
        description="List of node names to interrupt before (for human-in-the-loop)",
    )

    interrupt_after: Optional[List[str]] = Field(
        default=None,
        description="List of node names to interrupt after (for human-in-the-loop)",
    )

    @field_validator("triggers")
    @classmethod
    def validate_triggers(cls, v: List[str]) -> List[str]:
        """Validate trigger event kinds follow the required pattern."""
        import re

        pattern = r"^[a-zA-Z0-9_]+$"

        for trigger in v:
            if not re.match(pattern, trigger):
                raise ValueError(f"Trigger '{trigger}' must match pattern '{pattern}'")

        return v


class GraphOrchestrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Master configuration for multi-graph LangGraph orchestration.

    This model defines the overall orchestration setup, including which graphs
    to load and their runtime configurations.
    """

    graphs: List["GraphConfig"] = Field(
        min_length=1, description="List of LangGraph workflows to orchestrate"
    )

    slice_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description="Time slice duration in milliseconds (overrides global config)",
    )

    checkpointing_backend: Literal["memory", "sqlite", "postgres"] = Field(
        default="memory",
        description="Backend for LangGraph checkpoint persistence",
    )

    checkpointing_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific configuration for checkpointing (e.g., DB connection string)",
    )


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Configuration for a single LangGraph workflow in the orchestration."""

    path: str = Field(
        min_length=1,
        description="Path to directory containing LangGraph workflow Python module",
    )

    runtime: str = Field(
        min_length=1,
        description="Path to runtime configuration file (relative to graph path or absolute)",
    )

    graph_module: Optional[str] = Field(
        default=None,
        description="Python module path to import graph from (e.g., 'workflows.analysis')",
    )

    graph_factory: Optional[str] = Field(
        default="create_graph",
        description="Name of factory function that returns compiled LangGraph app",
    )
