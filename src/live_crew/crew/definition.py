"""Crew definition models for live-crew orchestration."""

from pydantic import BaseModel, ConfigDict, Field

from live_crew.core.dependencies import Dependency


class CrewDefinition(BaseModel):
    """Definition of a crew with its metadata and dependencies.

    Args:
        crew_id: Unique identifier for the crew (alphanumeric + underscore)
        triggers: List of event kinds that trigger this crew (non-empty)
        dependencies: List of dependencies that must be satisfied
        timeout_ms: Maximum execution time in milliseconds (positive)
        description: Human-readable description of the crew's purpose
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable like the original dataclass
        extra="forbid",  # Strict validation - reject unknown fields
    )

    crew_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Unique identifier for the crew",
    )
    triggers: list[str] = Field(
        min_length=1, description="List of event kinds that trigger this crew"
    )
    dependencies: list[Dependency] = Field(
        default_factory=list, description="List of dependencies that must be satisfied"
    )
    timeout_ms: int = Field(
        default=5000, gt=0, description="Maximum execution time in milliseconds"
    )
    description: str = Field(
        default="", description="Human-readable description of the crew's purpose"
    )
