"""Protocol definitions for dependency resolution in multi-crew orchestration."""

from typing import Protocol, List, Dict, Set
from dataclasses import dataclass

from live_crew.crewai_integration.wrapper import CrewAIWrapper


@dataclass
class ExecutionPhase:
    """Represents a phase of crew execution with parallel crews."""

    crews: List[str]
    phase_index: int

    def __len__(self) -> int:
        return len(self.crews)

    def __iter__(self):
        return iter(self.crews)


class DependencyResolverProtocol(Protocol):
    """Protocol for dependency resolution strategies in multi-crew orchestration.

    This protocol enables different dependency resolution algorithms while
    maintaining a consistent interface for the orchestrator.

    Implementations might include:
    - TopologicalDependencyResolver: Standard topological sorting
    - PriorityDependencyResolver: Priority-based with fallback strategies
    - CachingDependencyResolver: Memoized resolution for performance
    - DynamicDependencyResolver: Runtime dependency adjustment
    """

    def resolve_execution_order(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> List[ExecutionPhase]:
        """Resolve dependency-ordered execution phases for crews.

        Args:
            crews: List of crew wrappers with dependency configurations
            slice_idx: Current time slice index for slice-aware dependency resolution

        Returns:
            List of execution phases, each containing crews that can run in parallel

        Raises:
            DependencyError: If dependencies cannot be resolved
        """
        ...

    def get_dependency_depth(self, crews: List[CrewAIWrapper], slice_idx: int) -> int:
        """Get the maximum dependency depth for performance estimation.

        Args:
            crews: List of crew wrappers
            slice_idx: Current time slice index

        Returns:
            Maximum number of sequential dependency levels
        """
        ...

    def get_parallelism_factor(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> float:
        """Calculate parallelism factor for performance optimization.

        Args:
            crews: List of crew wrappers
            slice_idx: Current time slice index

        Returns:
            Average number of crews that can run in parallel (higher is better)
        """
        ...


class DependencyError(Exception):
    """Raised when dependency resolution fails due to cycles or unresolvable dependencies."""

    def __init__(self, message: str, graph: Dict[str, Set[str]] | None = None):
        super().__init__(message)
        self.graph = graph
