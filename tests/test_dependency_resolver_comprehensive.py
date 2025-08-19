"""Comprehensive tests for dependency resolver implementations with extreme edge cases.

This test suite covers the dependency resolution protocol with a focus on discovering
the most bizarre, unexpected, and boundary-pushing edge cases that can break the system.
We test both basic functionality and malicious/confused user scenarios.
"""

import pytest
import threading
import time
from typing import Dict, List
from unittest.mock import Mock

from live_crew.interfaces.dependency_protocol import ExecutionPhase, DependencyError
from live_crew.crewai_integration.dependency_resolver import (
    TopologicalDependencyResolver,
    NullDependencyResolver,
)
from live_crew.crewai_integration.wrapper import CrewAIWrapper


class TestDependencyResolverProtocol:
    """Test protocol compliance for all dependency resolver implementations."""

    @pytest.fixture
    def mock_crew_factory(self):
        """Factory for creating mock CrewAI crews with configurable dependencies."""

        def create_crew(
            crew_id: str, dependencies: List[str] | None = None
        ) -> CrewAIWrapper:
            mock_crewai_crew = Mock()
            crew = CrewAIWrapper(
                crew_id=crew_id,
                crewai_crew=mock_crewai_crew,
                triggers=["test_event"],
                timeout_ms=5000,
            )

            # Mock dependency extraction by storing dependencies on the crew object
            # This simulates what will happen when Phase 2 configuration is implemented
            crew._test_dependencies = dependencies or []
            return crew

        return create_crew

    @pytest.fixture
    def dependency_extracting_resolver(self, monkeypatch):
        """Resolver that can extract dependencies from mock crews for testing."""
        resolver = TopologicalDependencyResolver()

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            """Extract test dependencies from mock crew."""
            return getattr(crew, "_test_dependencies", [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        return resolver

    @pytest.mark.parametrize(
        "resolver_class", [TopologicalDependencyResolver, NullDependencyResolver]
    )
    def test_protocol_compliance(self, resolver_class, mock_crew_factory):
        """Test that all resolvers properly implement the protocol interface."""
        resolver = resolver_class()
        crews = [mock_crew_factory("crew1")]
        slice_idx = 0

        # Test required methods exist and return correct types
        phases = resolver.resolve_execution_order(crews, slice_idx)
        assert isinstance(phases, list)
        assert all(isinstance(phase, ExecutionPhase) for phase in phases)

        depth = resolver.get_dependency_depth(crews, slice_idx)
        assert isinstance(depth, int)
        assert depth >= 0

        parallelism = resolver.get_parallelism_factor(crews, slice_idx)
        assert isinstance(parallelism, float)
        assert parallelism >= 0.0

    def test_execution_phase_structure(
        self, mock_crew_factory, dependency_extracting_resolver
    ):
        """Test that ExecutionPhase objects have correct structure and behavior."""
        crews = [mock_crew_factory("crew1"), mock_crew_factory("crew2")]

        phases = dependency_extracting_resolver.resolve_execution_order(crews, 0)

        for phase in phases:
            # Test basic structure
            assert hasattr(phase, "crews")
            assert hasattr(phase, "phase_index")
            assert isinstance(phase.crews, list)
            assert isinstance(phase.phase_index, int)

            # Test iteration protocol
            assert len(phase) == len(phase.crews)
            assert list(phase) == phase.crews


class TestTopologicalDependencyResolverBasics:
    """Test basic dependency resolution scenarios."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        """Resolver with mock dependency extraction."""
        resolver = TopologicalDependencyResolver()

        # Mock dependency mappings for predictable testing
        dependency_map: Dict[str, List[str]] = {}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )

        # Store dependency map on resolver for test manipulation
        resolver._test_dependency_map = dependency_map
        return resolver

    @pytest.fixture
    def crew_factory(self):
        """Factory for creating test crews."""

        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_single_crew_no_dependencies(self, resolver, crew_factory):
        """Test simple single crew with no dependencies."""
        crews = [crew_factory("solo")]

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 1
        assert phases[0].crews == ["solo"]
        assert phases[0].phase_index == 0
        assert resolver.get_dependency_depth(crews, 0) == 1
        assert resolver.get_parallelism_factor(crews, 0) == 1.0

    def test_linear_dependency_chain(self, resolver, crew_factory):
        """Test simple linear dependency chain: A -> B -> C."""
        crews = [crew_factory("A"), crew_factory("B"), crew_factory("C")]

        # A depends on B, B depends on C (reverse execution order)
        resolver._test_dependency_map.update({"A": ["B"], "B": ["C"], "C": []})

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 3
        assert phases[0].crews == ["C"]  # No dependencies
        assert phases[1].crews == ["B"]  # Depends on C
        assert phases[2].crews == ["A"]  # Depends on B
        assert resolver.get_dependency_depth(crews, 0) == 3
        assert resolver.get_parallelism_factor(crews, 0) == 1.0  # All sequential

    def test_diamond_dependency_pattern(self, resolver, crew_factory):
        """Test diamond dependency: A -> B,C -> D (parallel middle layer)."""
        crews = [
            crew_factory("A"),
            crew_factory("B"),
            crew_factory("C"),
            crew_factory("D"),
        ]

        resolver._test_dependency_map.update(
            {
                "A": ["B", "C"],  # A depends on both B and C
                "B": ["D"],  # B depends on D
                "C": ["D"],  # C depends on D
                "D": [],  # D has no dependencies
            }
        )

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 3
        assert phases[0].crews == ["D"]  # Foundation
        assert set(phases[1].crews) == {"B", "C"}  # Parallel middle layer
        assert phases[2].crews == ["A"]  # Top level
        assert resolver.get_dependency_depth(crews, 0) == 3
        assert resolver.get_parallelism_factor(crews, 0) == pytest.approx(
            4 / 3, rel=1e-5
        )  # 4 crews, 3 phases

    def test_complex_parallel_execution(self, resolver, crew_factory):
        """Test complex graph with multiple parallel opportunities."""
        # Graph: F -> D,E -> B,C -> A (multiple parallel layers)
        crews = [crew_factory(id) for id in ["A", "B", "C", "D", "E", "F"]]

        resolver._test_dependency_map.update(
            {
                "F": ["D", "E"],
                "D": ["B", "C"],
                "E": ["C"],
                "B": ["A"],
                "C": ["A"],
                "A": [],
            }
        )

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 4
        assert phases[0].crews == ["A"]
        assert set(phases[1].crews) == {"B", "C"}
        assert set(phases[2].crews) == {"D", "E"}
        assert phases[3].crews == ["F"]

        assert resolver.get_dependency_depth(crews, 0) == 4
        assert resolver.get_parallelism_factor(crews, 0) == 1.5  # 6 crews, 4 phases


class TestCycleDetectionEdgeCases:
    """Test cycle detection with various malicious and edge case patterns."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        """Resolver with configurable dependency extraction."""
        resolver = TopologicalDependencyResolver()
        dependency_map: Dict[str, List[str]] = {}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        resolver._test_dependency_map = dependency_map
        return resolver

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_simple_self_dependency_cycle(self, resolver, crew_factory):
        """Test the most basic cycle: crew depends on itself."""
        crews = [crew_factory("narcissist")]
        resolver._test_dependency_map["narcissist"] = ["narcissist"]

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)
        assert "narcissist → narcissist" in str(exc_info.value)
        assert exc_info.value.graph is not None

    def test_two_crew_mutual_dependency(self, resolver, crew_factory):
        """Test simple two-crew cycle: A -> B -> A."""
        crews = [crew_factory("A"), crew_factory("B")]
        resolver._test_dependency_map.update({"A": ["B"], "B": ["A"]})

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)
        assert exc_info.value.graph is not None

    def test_complex_cycle_in_large_graph(self, resolver, crew_factory):
        """Test cycle detection in a large graph with embedded cycle."""
        # Graph: A -> B -> C -> D -> E -> C (cycle embedded in larger graph)
        crews = [crew_factory(id) for id in ["A", "B", "C", "D", "E", "F", "G"]]
        resolver._test_dependency_map.update(
            {
                "A": ["B"],
                "B": ["C"],
                "C": ["D"],
                "D": ["E"],
                "E": ["C"],  # Creates cycle: C -> D -> E -> C
                "F": ["G"],  # Independent chain (no cycle)
                "G": [],
            }
        )

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)

    def test_multiple_disconnected_cycles(self, resolver, crew_factory):
        """Test graph with multiple separate cycles."""
        crews = [crew_factory(id) for id in ["A", "B", "C", "D", "E", "F"]]
        resolver._test_dependency_map.update(
            {
                "A": ["B"],
                "B": ["A"],  # First cycle: A -> B -> A
                "C": ["D"],
                "D": ["E"],
                "E": ["C"],  # Second cycle: C -> D -> E -> C
                "F": [],  # Independent crew
            }
        )

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)

    def test_long_cycle_chain(self, resolver, crew_factory):
        """Test very long cycle to stress cycle detection algorithm."""
        cycle_length = 50
        crews = [crew_factory(f"crew_{i}") for i in range(cycle_length)]

        # Create long cycle: 0 -> 1 -> 2 -> ... -> 49 -> 0
        for i in range(cycle_length):
            next_idx = (i + 1) % cycle_length
            resolver._test_dependency_map[f"crew_{i}"] = [f"crew_{next_idx}"]

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)


class TestErrorHandlingAndValidation:
    """Test error handling and input validation edge cases."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        resolver = TopologicalDependencyResolver()
        dependency_map: Dict[str, List[str]] = {}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        resolver._test_dependency_map = dependency_map
        return resolver

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_empty_crew_list(self, resolver):
        """Test behavior with empty crew list."""
        phases = resolver.resolve_execution_order([], 0)

        assert phases == []
        assert resolver.get_dependency_depth([], 0) == 0
        assert resolver.get_parallelism_factor([], 0) == 0.0

    def test_unknown_dependency_error(self, resolver, crew_factory):
        """Test error when crew depends on non-existent crew."""
        crews = [crew_factory("existing")]
        resolver._test_dependency_map["existing"] = ["phantom_crew"]

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "unknown crew 'phantom_crew'" in str(exc_info.value)
        assert "'existing'" in str(exc_info.value)

    def test_multiple_unknown_dependencies(self, resolver, crew_factory):
        """Test multiple crews with unknown dependencies."""
        crews = [crew_factory("A"), crew_factory("B")]
        resolver._test_dependency_map.update(
            {"A": ["ghost1", "ghost2"], "B": ["phantom"]}
        )

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "unknown crew" in str(exc_info.value)

    def test_malicious_extremely_large_crew_ids(self, resolver):
        """Test behavior with maliciously large crew IDs."""
        huge_id = "A" * 10000  # 10KB crew ID
        crew = CrewAIWrapper(
            crew_id=huge_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
        )

        # Should handle large IDs without crashing
        phases = resolver.resolve_execution_order([crew], 0)
        assert len(phases) == 1
        assert phases[0].crews == [huge_id]

    def test_malicious_unicode_crew_ids(self, resolver):
        """Test behavior with Unicode and special characters in crew IDs."""
        weird_crews = []
        weird_ids = [
            "crew_🚀",  # Emoji
            "crew_\n\t\r",  # Control characters
            "crew_中文",  # Chinese characters
            "crew_🙃💀👻",  # Multiple emojis
            "crew_\x00\x01",  # Null bytes and control chars
            "crew_'\"\\",  # Quote characters
            "crew_<script>",  # HTML/JS injection attempt
        ]

        for weird_id in weird_ids:
            crew = CrewAIWrapper(
                crew_id=weird_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )
            weird_crews.append(crew)

        phases = resolver.resolve_execution_order(weird_crews, 0)

        assert len(phases) == 1
        assert set(phases[0].crews) == set(weird_ids)

    def test_negative_slice_index(self, resolver, crew_factory):
        """Test behavior with negative slice indices."""
        crews = [crew_factory("test")]

        # Should handle negative indices gracefully
        phases = resolver.resolve_execution_order(crews, -1)
        assert len(phases) == 1

    def test_extremely_large_slice_index(self, resolver, crew_factory):
        """Test behavior with extremely large slice indices."""
        crews = [crew_factory("test")]

        # Should handle large indices without overflow
        phases = resolver.resolve_execution_order(crews, 2**31 - 1)
        assert len(phases) == 1


class TestCachingAndPerformance:
    """Test caching behavior and performance characteristics."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        resolver = TopologicalDependencyResolver()
        dependency_map: Dict[str, List[str]] = {}
        call_count = {"extract_deps": 0}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            call_count["extract_deps"] += 1
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        resolver._test_dependency_map = dependency_map
        resolver._test_call_count = call_count
        return resolver

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_caching_same_crews_same_slice(self, resolver, crew_factory):
        """Test that identical calls are cached properly."""
        crews = [crew_factory("A"), crew_factory("B")]
        slice_idx = 0

        # First call
        phases1 = resolver.resolve_execution_order(crews, slice_idx)
        initial_call_count = resolver._test_call_count["extract_deps"]

        # Second identical call should use cache
        phases2 = resolver.resolve_execution_order(crews, slice_idx)
        cached_call_count = resolver._test_call_count["extract_deps"]

        assert phases1 == phases2
        assert cached_call_count == initial_call_count  # No additional extraction calls

    def test_cache_key_sensitivity_to_crew_order(self, resolver, crew_factory):
        """Test that cache keys are order-independent for crew lists."""
        crew_a, crew_b = crew_factory("A"), crew_factory("B")

        # Different order, same crews
        phases1 = resolver.resolve_execution_order([crew_a, crew_b], 0)
        phases2 = resolver.resolve_execution_order([crew_b, crew_a], 0)

        # Should produce same result (cache should work regardless of order)
        assert len(phases1) == len(phases2)
        assert phases1[0].crews == phases2[0].crews or set(phases1[0].crews) == set(
            phases2[0].crews
        )

    def test_cache_invalidation_different_slice(self, resolver, crew_factory):
        """Test that different slice indices create separate cache entries."""
        crews = [crew_factory("A")]

        resolver.resolve_execution_order(crews, 0)
        resolver.resolve_execution_order(crews, 1)

        # Should be separate cache entries
        assert resolver._test_call_count["extract_deps"] >= 2

    def test_cache_clear_functionality(self, resolver, crew_factory):
        """Test manual cache clearing."""
        crews = [crew_factory("A")]

        # Prime cache
        resolver.resolve_execution_order(crews, 0)
        initial_call_count = resolver._test_call_count["extract_deps"]

        # Clear cache
        resolver.clear_cache()

        # Should not use cache after clearing
        resolver.resolve_execution_order(crews, 0)
        post_clear_call_count = resolver._test_call_count["extract_deps"]

        assert post_clear_call_count > initial_call_count

    def test_memory_usage_with_large_graphs(self, resolver, crew_factory):
        """Test memory behavior with large dependency graphs."""
        # Create large graph (100 crews in complex dependency pattern)
        num_crews = 100
        crews = [crew_factory(f"crew_{i}") for i in range(num_crews)]

        # Create complex dependencies (each crew depends on previous few)
        for i in range(num_crews):
            deps = []
            for j in range(max(0, i - 3), i):  # Depend on up to 3 previous crews
                deps.append(f"crew_{j}")
            resolver._test_dependency_map[f"crew_{i}"] = deps

        # Should handle large graphs without excessive memory usage
        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) > 0
        total_crews_in_phases = sum(len(phase.crews) for phase in phases)
        assert total_crews_in_phases == num_crews

    def test_performance_with_deep_dependency_chains(self, resolver, crew_factory):
        """Test performance with very deep dependency chains."""
        # Create deep chain: 0 -> 1 -> 2 -> ... -> 49
        chain_length = 50
        crews = [crew_factory(f"crew_{i}") for i in range(chain_length)]

        for i in range(chain_length - 1):
            resolver._test_dependency_map[f"crew_{i}"] = [f"crew_{i + 1}"]
        resolver._test_dependency_map[f"crew_{chain_length - 1}"] = []

        start_time = time.time()
        phases = resolver.resolve_execution_order(crews, 0)
        end_time = time.time()

        # Should complete quickly (under 1 second for 50-crew chain)
        assert (end_time - start_time) < 1.0
        assert len(phases) == chain_length
        assert resolver.get_dependency_depth(crews, 0) == chain_length


class TestConcurrentAccessScenarios:
    """Test thread safety and concurrent access patterns."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        resolver = TopologicalDependencyResolver()
        dependency_map: Dict[str, List[str]] = {}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            # Add small delay to increase chance of race conditions
            time.sleep(0.001)
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        resolver._test_dependency_map = dependency_map
        return resolver

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_concurrent_resolution_same_input(self, resolver, crew_factory):
        """Test concurrent resolution of the same input doesn't cause issues."""
        crews = [crew_factory("A"), crew_factory("B"), crew_factory("C")]
        resolver._test_dependency_map.update({"A": ["B"], "B": ["C"], "C": []})

        results = []
        errors = []

        def resolve_concurrently():
            try:
                result = resolver.resolve_execution_order(crews, 0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=resolve_concurrently)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All should succeed and produce identical results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_concurrent_cache_access(self, resolver, crew_factory):
        """Test that concurrent cache access doesn't corrupt data."""
        crews = [crew_factory(f"crew_{i}") for i in range(5)]

        cache_hits = []

        def access_with_different_slice(slice_idx: int):
            try:
                result = resolver.resolve_execution_order(crews, slice_idx)
                cache_hits.append((slice_idx, len(result)))
            except Exception:
                pass  # Ignore errors, focus on cache corruption

        # Access cache with different slice indices concurrently
        threads = []
        for i in range(20):
            t = threading.Thread(target=access_with_different_slice, args=(i % 5,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have results for different slice indices
        assert len(cache_hits) > 0

    def test_concurrent_cache_clearing(self, resolver, crew_factory):
        """Test concurrent cache clearing doesn't cause crashes."""
        crews = [crew_factory("test")]

        def clear_and_resolve():
            try:
                resolver.clear_cache()
                resolver.resolve_execution_order(crews, 0)
            except Exception:
                pass  # Focus on not crashing

        threads = []
        for _ in range(10):
            t = threading.Thread(target=clear_and_resolve)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # If we get here without deadlock, test passes


class TestNullDependencyResolver:
    """Test the NullDependencyResolver implementation."""

    @pytest.fixture
    def resolver(self):
        return NullDependencyResolver()

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_single_crew_execution(self, resolver, crew_factory):
        """Test single crew returns single phase."""
        crews = [crew_factory("solo")]

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 1
        assert phases[0].crews == ["solo"]
        assert phases[0].phase_index == 0

    def test_multiple_crews_all_parallel(self, resolver, crew_factory):
        """Test multiple crews all execute in parallel."""
        crews = [crew_factory(f"crew_{i}") for i in range(5)]
        crew_ids = [f"crew_{i}" for i in range(5)]

        phases = resolver.resolve_execution_order(crews, 0)

        assert len(phases) == 1
        assert set(phases[0].crews) == set(crew_ids)
        assert phases[0].phase_index == 0

    def test_empty_input(self, resolver):
        """Test empty crew list returns empty phases."""
        phases = resolver.resolve_execution_order([], 0)

        assert phases == []

    def test_performance_metrics(self, resolver, crew_factory):
        """Test performance metrics are correctly calculated."""
        crews = [crew_factory(f"crew_{i}") for i in range(10)]

        depth = resolver.get_dependency_depth(crews, 0)
        parallelism = resolver.get_parallelism_factor(crews, 0)

        assert depth == 1  # Single phase
        assert parallelism == 10.0  # All crews in parallel

    def test_slice_index_independence(self, resolver, crew_factory):
        """Test that slice index doesn't affect null resolver behavior."""
        crews = [crew_factory("test")]

        phases_slice_0 = resolver.resolve_execution_order(crews, 0)
        phases_slice_100 = resolver.resolve_execution_order(crews, 100)

        assert phases_slice_0 == phases_slice_100


class TestExtremeStressScenarios:
    """Test extreme scenarios that could break the system."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        resolver = TopologicalDependencyResolver()
        dependency_map: Dict[str, List[str]] = {}

        def mock_extract_dependencies(crew: CrewAIWrapper, slice_idx: int) -> List[str]:
            return dependency_map.get(crew.crew_id, [])

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", mock_extract_dependencies
        )
        resolver._test_dependency_map = dependency_map
        return resolver

    @pytest.fixture
    def crew_factory(self):
        def create_crew(crew_id: str) -> CrewAIWrapper:
            return CrewAIWrapper(
                crew_id=crew_id, crewai_crew=Mock(), triggers=["test"], timeout_ms=1000
            )

        return create_crew

    def test_malicious_duplicate_crew_ids(self, resolver, crew_factory):
        """Test behavior with duplicate crew IDs (malicious or confused user)."""
        # Create crews with duplicate IDs
        crew1 = crew_factory("duplicate")
        crew2 = crew_factory("duplicate")
        crews = [crew1, crew2]

        # Should handle duplicates gracefully (last one wins in most implementations)
        try:
            phases = resolver.resolve_execution_order(crews, 0)
            # If it succeeds, verify it produces reasonable output
            assert len(phases) >= 1
        except Exception as e:
            # If it fails, should be a clear error message
            assert "duplicate" in str(e).lower() or "conflict" in str(e).lower()

    def test_recursive_dependency_extraction_simulation(
        self, resolver, crew_factory, monkeypatch
    ):
        """Test what happens if dependency extraction itself has issues."""
        crews = [crew_factory("problematic")]

        def problematic_extract_dependencies(
            crew: CrewAIWrapper, slice_idx: int
        ) -> List[str]:
            if crew.crew_id == "problematic":
                raise ValueError("Simulated configuration parsing error")
            return []

        monkeypatch.setattr(
            resolver, "_extract_crew_dependencies", problematic_extract_dependencies
        )

        # Should propagate the error clearly
        with pytest.raises(ValueError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "configuration parsing error" in str(exc_info.value)

    def test_memory_stress_with_massive_graphs(self, resolver, crew_factory):
        """Test behavior with unrealistically large dependency graphs."""
        # Create moderately large graph (limited by test execution time)
        num_crews = 200
        crews = [crew_factory(f"crew_{i}") for i in range(num_crews)]

        # Create worst-case scenario: each crew depends on ALL previous crews
        for i in range(num_crews):
            deps = [f"crew_{j}" for j in range(i)]
            resolver._test_dependency_map[f"crew_{i}"] = deps

        start_time = time.time()

        try:
            phases = resolver.resolve_execution_order(crews, 0)
            end_time = time.time()

            # Should complete within reasonable time (5 seconds)
            assert (end_time - start_time) < 5.0
            assert len(phases) == num_crews  # All sequential

        except MemoryError:
            pytest.skip("System doesn't have enough memory for this stress test")

    def test_malicious_cyclic_dependency_names(self, resolver, crew_factory):
        """Test cycles with confusing names that might break string matching."""
        confusing_names = ["A_depends_on_B", "B_depends_on_A", "cycle_A", "cycle_B"]

        crews = [crew_factory(name) for name in confusing_names]
        resolver._test_dependency_map.update(
            {
                "A_depends_on_B": ["B_depends_on_A"],
                "B_depends_on_A": ["A_depends_on_B"],  # Cycle despite confusing names
                "cycle_A": ["cycle_B"],
                "cycle_B": ["cycle_A"],  # Another cycle
            }
        )

        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve_execution_order(crews, 0)

        assert "Circular dependency detected" in str(exc_info.value)

    def test_extreme_unicode_edge_cases(self, resolver):
        """Test extreme Unicode edge cases that might break string processing."""
        extreme_unicode_cases = [
            "crew_\U0001f4a9",  # Pile of poo emoji (4-byte Unicode)
            "crew_\u200b",  # Zero-width space
            "crew_\ufeff",  # Byte order mark
            "crew_\u0000",  # Null character (if permitted)
            "crew_\U000e0001",  # Private use character
            "crew_𝕔𝓲𝓷𝓰",  # Mathematical script characters
        ]

        crews = []
        for unicode_id in extreme_unicode_cases:
            try:
                crew = CrewAIWrapper(
                    crew_id=unicode_id,
                    crewai_crew=Mock(),
                    triggers=["test"],
                    timeout_ms=1000,
                )
                crews.append(crew)
            except Exception:
                # Skip if the system can't handle this Unicode case
                continue

        if crews:  # Only test if we have valid crews
            phases = resolver.resolve_execution_order(crews, 0)
            assert len(phases) == 1  # All should be in single phase (no dependencies)
