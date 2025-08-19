"""Pydantic models for CrewAI integration configuration."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

from live_crew.core.dependencies import Dependency


class CrewRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Runtime configuration for CrewAI crew integration with live-crew orchestration.

    This model defines how a standard CrewAI crew integrates with live-crew's
    event-driven orchestration system, including triggers, dependencies, and
    execution parameters.
    """

    crew: str = Field(
        min_length=1,
        description="Unique identifier for this crew within live-crew orchestration",
    )

    triggers: List[str] = Field(
        min_length=1,
        description="List of event kinds that trigger this crew's execution",
    )

    needs: Optional[List[Dependency]] = Field(
        default=None,
        description="Dependencies that must be satisfied before crew execution",
    )

    wait_policy: Literal["any", "all", "none"] = Field(
        default="none", description="Policy for handling dependency satisfaction"
    )

    timeout_ms: int = Field(
        default=5000,
        gt=0,
        le=300000,  # Max 5 minutes
        description="Maximum execution time for crew processing in milliseconds",
    )

    slice_stride: int = Field(
        default=1, ge=1, description="Execute crew every N slices (1 = every slice)"
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


class CrewOrchestrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Master configuration for multi-crew CrewAI orchestration.

    This model defines the overall orchestration setup, including which crews
    to load and their runtime configurations.
    """

    crews: List["CrewConfig"] = Field(
        min_length=1, description="List of CrewAI crews to orchestrate"
    )

    slice_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description="Time slice duration in milliseconds (overrides global config)",
    )


class CrewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Configuration for a single CrewAI crew in the orchestration."""

    path: str = Field(
        min_length=1, description="Path to directory containing CrewAI crew files"
    )

    runtime: str = Field(
        min_length=1,
        description="Path to runtime configuration file (relative to crew path or absolute)",
    )
