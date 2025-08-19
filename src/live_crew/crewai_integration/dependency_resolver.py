"""Topological sorting implementation of dependency resolution for multi-crew orchestration.

This module provides a production-ready implementation of the DependencyResolverProtocol
using topological sorting with cycle detection and parallel execution optimization.
"""

from typing import Dict, Set, List, Tuple

from live_crew.interfaces.dependency_protocol import (
    DependencyResolverProtocol,
    ExecutionPhase,
    DependencyError,
)
from live_crew.crewai_integration.wrapper import CrewAIWrapper


class TopologicalDependencyResolver(DependencyResolverProtocol):
    """Topological sorting implementation with cycle detection and parallel execution planning.

    This resolver builds dependency graphs from crew configurations, detects circular
    dependencies, and produces execution phases that maximize parallelism while
    respecting dependency constraints.

    Example:
        resolver = TopologicalDependencyResolver()
        phases = resolver.resolve_execution_order(crew_wrappers, slice_idx=0)
        for phase in phases:
            # Execute crews in this phase concurrently
            await execute_crew_phase(phase.crews)
    """

    def __init__(self):
        """Initialize the dependency resolver."""
        self._dependency_cache: Dict[Tuple[List[str], int], List[ExecutionPhase]] = {}

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
            DependencyError: If circular dependencies are detected or dependencies are unresolvable
        """
        # Create cache key for memoization
        crew_ids = tuple(sorted(crew.crew_id for crew in crews))
        cache_key = (crew_ids, slice_idx)

        if cache_key in self._dependency_cache:
            return self._dependency_cache[cache_key]

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(crews, slice_idx)

        # Detect cycles before attempting resolution
        cycle_path = self._detect_cycle(dependency_graph)
        if cycle_path:
            raise DependencyError(
                f"Circular dependency detected: {' → '.join(cycle_path + [cycle_path[0]])}",
                graph=dependency_graph,
            )

        # Perform topological sort with parallel phase detection
        execution_phases = self._topological_sort_parallel(dependency_graph)

        # Cache result for future use
        self._dependency_cache[cache_key] = execution_phases

        return execution_phases

    def get_dependency_depth(self, crews: List[CrewAIWrapper], slice_idx: int) -> int:
        """Get the maximum dependency depth for performance estimation."""
        phases = self.resolve_execution_order(crews, slice_idx)
        return len(phases)

    def get_parallelism_factor(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> float:
        """Calculate parallelism factor for performance optimization."""
        phases = self.resolve_execution_order(crews, slice_idx)
        if not phases:
            return 0.0

        total_crews = sum(len(phase) for phase in phases)
        return total_crews / len(phases)

    def _build_dependency_graph(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from crew configurations.

        Args:
            crews: List of crew wrappers with dependency info
            slice_idx: Current time slice for slice-aware dependencies

        Returns:
            Dictionary mapping crew_id -> set of crew_ids it depends on
        """
        graph: Dict[str, Set[str]] = {}
        crew_by_id = {crew.crew_id: crew for crew in crews}

        # Initialize graph with all crews
        for crew in crews:
            graph[crew.crew_id] = set()

        # Build dependency edges
        for crew in crews:
            dependencies = self._extract_crew_dependencies(crew, slice_idx)

            for dep_crew_id in dependencies:
                if dep_crew_id not in crew_by_id:
                    raise DependencyError(
                        f"Crew '{crew.crew_id}' depends on unknown crew '{dep_crew_id}'"
                    )
                graph[crew.crew_id].add(dep_crew_id)

        return graph

    def _extract_crew_dependencies(
        self, crew: CrewAIWrapper, slice_idx: int
    ) -> List[str]:
        """Extract crew dependencies from wrapper configuration.

        Note: This is a placeholder implementation. In the full implementation,
        this would read from the crew's runtime configuration once we enhance
        the configuration models in Phase 2.

        Args:
            crew: Crew wrapper to extract dependencies from
            slice_idx: Current time slice for offset-based dependencies

        Returns:
            List of crew IDs this crew depends on
        """
        # TODO: Once CrewRuntimeConfig is enhanced with dependencies in Phase 2,
        # this will read from crew.config.dependencies and resolve offset-based deps

        # For now, return empty dependencies to support current single-crew scenarios
        return []

    def _detect_cycle(self, graph: Dict[str, Set[str]]) -> List[str] | None:
        """Detect cycles in the dependency graph using DFS.

        Args:
            graph: Dependency graph to check

        Returns:
            List representing the cycle path if found, None if acyclic
        """
        WHITE = 0  # Unvisited
        GRAY = 1  # Currently being processed
        BLACK = 2  # Completed processing

        colors = {node: WHITE for node in graph}
        parent = {node: None for node in graph}

        def dfs_visit(node: str) -> List[str] | None:
            colors[node] = GRAY

            for neighbor in graph[node]:
                if colors[neighbor] == GRAY:
                    # Found back edge - cycle detected
                    # Reconstruct cycle path
                    cycle = []
                    current = node
                    while True:
                        cycle.append(current)
                        if current == neighbor:
                            break
                        current = (
                            parent[current] or neighbor
                        )  # Fallback to neighbor if parent is None
                    return cycle

                elif colors[neighbor] == WHITE:
                    parent[neighbor] = node
                    cycle = dfs_visit(neighbor)
                    if cycle:
                        return cycle

            colors[node] = BLACK
            return None

        # Check all components for cycles
        for node in graph:
            if colors[node] == WHITE:
                cycle = dfs_visit(node)
                if cycle:
                    return cycle

        return None

    def _topological_sort_parallel(
        self, graph: Dict[str, Set[str]]
    ) -> List[ExecutionPhase]:
        """Perform topological sort optimized for parallel execution phases.

        This algorithm groups crews into execution phases where all crews in a
        phase can execute concurrently without violating dependencies.

        Args:
            graph: Dependency graph (crew_id -> dependencies)

        Returns:
            List of execution phases for optimal parallel processing

        Raises:
            DependencyError: If unresolvable dependencies remain
        """
        execution_phases: List[ExecutionPhase] = []

        # Calculate in-degrees (number of dependencies for each crew)
        in_degree = {node: len(dependencies) for node, dependencies in graph.items()}
        remaining_graph = {node: deps.copy() for node, deps in graph.items()}

        phase_index = 0

        while in_degree:
            # Find all crews with no remaining dependencies (ready to execute)
            ready_crews = [crew for crew, degree in in_degree.items() if degree == 0]

            if not ready_crews:
                # No progress possible - unresolvable dependencies
                remaining_crews = list(in_degree.keys())
                remaining_deps = {
                    crew: list(deps) for crew, deps in remaining_graph.items() if deps
                }
                raise DependencyError(
                    f"Unresolvable dependencies detected. Remaining crews: {remaining_crews}. "
                    f"Remaining dependencies: {remaining_deps}"
                )

            # Create execution phase for ready crews
            execution_phases.append(ExecutionPhase(ready_crews, phase_index))
            phase_index += 1

            # Remove completed crews and update dependencies
            for completed_crew in ready_crews:
                del in_degree[completed_crew]

                # Update in-degrees for crews that depended on the completed crew
                for other_crew in list(in_degree.keys()):
                    if completed_crew in remaining_graph[other_crew]:
                        remaining_graph[other_crew].remove(completed_crew)
                        in_degree[other_crew] -= 1

        return execution_phases

    def clear_cache(self) -> None:
        """Clear the dependency resolution cache."""
        self._dependency_cache.clear()


class NullDependencyResolver(DependencyResolverProtocol):
    """No-op dependency resolver for single-crew scenarios.

    This resolver treats all crews as independent with no dependencies,
    allowing them all to execute in a single parallel phase.
    """

    def resolve_execution_order(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> List[ExecutionPhase]:
        """Return single phase with all crews for parallel execution."""
        if not crews:
            return []

        crew_ids = [crew.crew_id for crew in crews]
        return [ExecutionPhase(crew_ids, 0)]

    def get_dependency_depth(self, crews: List[CrewAIWrapper], slice_idx: int) -> int:
        """Return depth of 1 for single-phase execution."""
        return 1 if crews else 0

    def get_parallelism_factor(
        self, crews: List[CrewAIWrapper], slice_idx: int
    ) -> float:
        """Return crew count since all execute in parallel."""
        return float(len(crews))
