"""Crew definition models for live-crew orchestration."""

from dataclasses import dataclass

from live_crew.core.dependencies import Dependency


@dataclass(frozen=True)
class CrewDefinition:
    """Definition of a crew with its metadata and dependencies.

    Args:
        crew_id: Unique identifier for the crew
        triggers: List of event kinds that trigger this crew
        dependencies: List of dependencies that must be satisfied
        timeout_ms: Maximum execution time in milliseconds
        description: Human-readable description of the crew's purpose
    """

    crew_id: str
    triggers: list[str]
    dependencies: list[Dependency]
    timeout_ms: int = 5000  # 5 second default timeout
    description: str = ""

    def __post_init__(self) -> None:
        """Validate crew definition fields."""
        if not self.crew_id:
            raise ValueError("crew_id cannot be empty")
        if not self.triggers:
            raise ValueError("triggers cannot be empty")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
