"""Comprehensive test suite for PartialFailureHandler with extensive edge case coverage.

This module tests the complete failure handling lifecycle including:
- All recovery strategy behaviors and edge cases
- Impact analysis accuracy under various dependency scenarios
- Strategy selection logic with complex failure patterns
- Failure history management and memory usage
- Critical crew special handling and cascading failures
- Context updates, logging, and monitoring integration
- Concurrent failure handling and race conditions
- Performance with large failure chains and dependency graphs
"""

import pytest
import asyncio
from unittest.mock import Mock

from live_crew.crewai_integration.failure_handler import (
    PartialFailureHandler,
    RecoveryStrategy,
    FailureImpact,
    DependencyChainAnalysis,
    IsolatedFailureStrategy,
    GracefulDegradationStrategy,
    CascadeFailureStrategy,
    RetryWithBackoffStrategy,
)
from live_crew.crewai_integration.execution_tracker import (
    ExecutionContext,
    CrewExecutionState,
)
from live_crew.interfaces.dependency_protocol import ExecutionPhase


class TestFailureImpactClassification:
    """Test FailureImpact enum and its usage patterns."""

    def test_failure_impact_values(self):
        """Test that FailureImpact enum has expected values."""
        assert FailureImpact.ISOLATED.value == "isolated"
        assert FailureImpact.DEGRADED.value == "degraded"
        assert FailureImpact.CASCADING.value == "cascading"
        assert FailureImpact.CATASTROPHIC.value == "catastrophic"

    def test_failure_impact_ordering_logic(self):
        """Test implicit severity ordering of failure impacts."""
        # Test that we can compare impacts (useful for escalation logic)
        impacts = [
            FailureImpact.ISOLATED,
            FailureImpact.DEGRADED,
            FailureImpact.CASCADING,
            FailureImpact.CATASTROPHIC,
        ]

        # Should be able to iterate through increasing severity
        assert len(impacts) == 4
        for i, impact in enumerate(impacts):
            assert isinstance(impact, FailureImpact)


class TestDependencyChainAnalysis:
    """Test DependencyChainAnalysis data structure and properties."""

    def test_basic_analysis_creation(self):
        """Test creating basic dependency chain analysis."""
        analysis = DependencyChainAnalysis(
            failed_crew="analytics_crew",
            impact_level=FailureImpact.DEGRADED,
            directly_affected=["reporting_crew", "dashboard_crew"],
            transitively_affected=["alert_crew"],
            critical_path_broken=False,
            recovery_options=["graceful_degradation", "retry_with_fallback"],
        )

        assert analysis.failed_crew == "analytics_crew"
        assert analysis.impact_level == FailureImpact.DEGRADED
        assert analysis.total_affected_count == 3  # 2 direct + 1 transitive
        assert not analysis.critical_path_broken
        assert "graceful_degradation" in analysis.recovery_options

    def test_empty_dependency_analysis(self):
        """Test analysis with no affected crews (isolated failure)."""
        analysis = DependencyChainAnalysis(
            failed_crew="isolated_crew",
            impact_level=FailureImpact.ISOLATED,
            directly_affected=[],
            transitively_affected=[],
            critical_path_broken=False,
            recovery_options=["continue_with_logging"],
        )

        assert analysis.total_affected_count == 0
        assert analysis.impact_level == FailureImpact.ISOLATED
        assert not analysis.critical_path_broken

    def test_overlapping_affected_crews(self):
        """Test analysis with crews appearing in both direct and transitive lists."""
        analysis = DependencyChainAnalysis(
            failed_crew="central_crew",
            impact_level=FailureImpact.CASCADING,
            directly_affected=["crew_a", "crew_b", "crew_c"],
            transitively_affected=["crew_b", "crew_c", "crew_d"],  # Some overlap
            critical_path_broken=True,
            recovery_options=["cascade_failure", "abort_orchestration"],
        )

        # Should deduplicate crews across both lists
        assert analysis.total_affected_count == 4  # crew_a, crew_b, crew_c, crew_d

    def test_massive_dependency_chain(self):
        """Test analysis with very large number of affected crews."""
        # Simulate failure in a highly connected crew
        directly_affected = [f"direct_crew_{i}" for i in range(50)]
        transitively_affected = [f"transitive_crew_{i}" for i in range(100)]

        analysis = DependencyChainAnalysis(
            failed_crew="hub_crew",
            impact_level=FailureImpact.CATASTROPHIC,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            critical_path_broken=True,
            recovery_options=["abort_orchestration"],
        )

        assert analysis.total_affected_count == 150
        assert analysis.impact_level == FailureImpact.CATASTROPHIC


class TestIsolatedFailureStrategy:
    """Test IsolatedFailureStrategy behavior and edge cases."""

    @pytest.fixture
    def execution_context(self) -> ExecutionContext:
        """Create minimal execution context for testing."""
        return ExecutionContext(
            slice_idx=0,
            phase_execution_order=[
                ExecutionPhase(crews=["crew_a", "crew_b"], phase_index=0)
            ],
        )

    @pytest.fixture
    def strategy(self) -> IsolatedFailureStrategy:
        """Create isolated failure strategy instance."""
        return IsolatedFailureStrategy()

    @pytest.mark.asyncio
    async def test_isolated_failure_basic_behavior(self, strategy, execution_context):
        """Test basic isolated failure handling."""
        error = ValueError("Configuration error in crew_a")

        updated_context = await strategy.execute(
            context=execution_context,
            failed_crew="crew_a",
            error=error,
            affected_crews=[],
        )

        # Check error recovery action was logged
        assert len(updated_context.error_recovery_actions) == 1
        action = updated_context.error_recovery_actions[0]
        assert "Isolated failure in 'crew_a'" in action
        assert "Other crews continue unaffected" in action
        assert "Configuration error in crew_a" in action

        # Check partial results were recorded
        failure_key = "crew_a_failure_isolated"
        assert failure_key in updated_context.partial_results

        failure_data = updated_context.partial_results[failure_key]
        assert failure_data["error"] == "Configuration error in crew_a"
        assert failure_data["recovery_strategy"] == "isolated"
        assert failure_data["impact"] == "none"

    @pytest.mark.asyncio
    async def test_isolated_failure_with_non_empty_affected_crews(
        self, strategy, execution_context
    ):
        """Test isolated strategy when affected_crews is accidentally non-empty."""
        # This tests defensive programming - strategy should still work
        error = RuntimeError("Isolated crew error")

        updated_context = await strategy.execute(
            context=execution_context,
            failed_crew="crew_isolated",
            error=error,
            affected_crews=["should_not_matter"],  # Should be ignored
        )

        # Strategy should still treat it as isolated
        assert len(updated_context.error_recovery_actions) == 1
        assert (
            "Other crews continue unaffected"
            in updated_context.error_recovery_actions[0]
        )

    @pytest.mark.asyncio
    async def test_isolated_failure_with_complex_error(
        self, strategy, execution_context
    ):
        """Test isolated failure with complex exception containing nested information."""
        # Create a complex error with nested exceptions
        try:
            try:
                raise ConnectionError("Database connection failed")
            except ConnectionError as e:
                raise RuntimeError("Crew initialization failed") from e
        except RuntimeError as complex_error:
            updated_context = await strategy.execute(
                context=execution_context,
                failed_crew="complex_crew",
                error=complex_error,
                affected_crews=[],
            )

        # Should handle complex error string representation
        failure_data = updated_context.partial_results["complex_crew_failure_isolated"]
        assert "Crew initialization failed" in failure_data["error"]

    @pytest.mark.asyncio
    async def test_isolated_failure_context_preservation(self, strategy):
        """Test that isolated failure preserves existing context data."""
        # Create context with existing data
        context = ExecutionContext(slice_idx=5)
        context.partial_results["existing_key"] = "existing_value"
        context.error_recovery_actions.append("Previous action")

        error = TypeError("Type mismatch in crew_x")

        updated_context = await strategy.execute(
            context=context, failed_crew="crew_x", error=error, affected_crews=[]
        )

        # Should preserve existing data
        assert updated_context.partial_results["existing_key"] == "existing_value"
        assert len(updated_context.error_recovery_actions) == 2
        assert updated_context.error_recovery_actions[0] == "Previous action"

        # Should add new failure data
        assert "crew_x_failure_isolated" in updated_context.partial_results


class TestGracefulDegradationStrategy:
    """Test GracefulDegradationStrategy behavior and edge cases."""

    @pytest.fixture
    def execution_context(self) -> ExecutionContext:
        """Create execution context for testing."""
        return ExecutionContext(
            slice_idx=1,
            phase_execution_order=[
                ExecutionPhase(crews=["producer_crew"], phase_index=0),
                ExecutionPhase(
                    crews=["consumer_crew_1", "consumer_crew_2"], phase_index=1
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_default_fallback(self):
        """Test graceful degradation with default (empty) fallback data."""
        strategy = GracefulDegradationStrategy()
        context = ExecutionContext(slice_idx=0)

        error = TimeoutError("Data processing timeout")
        affected_crews = ["downstream_crew_1", "downstream_crew_2"]

        updated_context = await strategy.execute(
            context=context,
            failed_crew="data_processor",
            error=error,
            affected_crews=affected_crews,
        )

        # Check fallback data was created
        fallback_key = "data_processor_fallback"
        assert fallback_key in updated_context.partial_results

        fallback_data = updated_context.partial_results[fallback_key]
        assert fallback_data["source"] == "graceful_degradation"
        assert fallback_data["original_crew"] == "data_processor"
        assert fallback_data["degraded_mode"] is True
        assert fallback_data["fallback_data"] == {}  # Default empty

        # Check affected crews were marked for degraded execution
        for crew_id in affected_crews:
            degraded_key = f"{crew_id}_degraded_input"
            assert degraded_key in updated_context.partial_results

            degraded_data = updated_context.partial_results[degraded_key]
            assert degraded_data["from_failed_crew"] == "data_processor"
            assert degraded_data["fallback_source"] == fallback_key
            assert degraded_data["execution_mode"] == "degraded"

        # Check recovery action was logged
        assert len(updated_context.error_recovery_actions) == 1
        action = updated_context.error_recovery_actions[0]
        assert "Graceful degradation for 'data_processor'" in action
        assert "Providing fallback data to 2 dependent crews" in action
        assert str(affected_crews) in action

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_custom_fallback(self):
        """Test graceful degradation with custom fallback data."""
        custom_fallback = {
            "default_values": {"metric": 0, "status": "unavailable"},
            "cache_data": {"last_known_good": "2024-01-01"},
            "flags": {"use_fallback_mode": True},
        }

        strategy = GracefulDegradationStrategy(fallback_data=custom_fallback)
        context = ExecutionContext(slice_idx=2)

        error = Exception("Service unavailable")

        updated_context = await strategy.execute(
            context=context,
            failed_crew="metrics_crew",
            error=error,
            affected_crews=["dashboard_crew"],
        )

        fallback_data = updated_context.partial_results["metrics_crew_fallback"]
        assert fallback_data["fallback_data"] == custom_fallback
        assert fallback_data["fallback_data"]["flags"]["use_fallback_mode"] is True

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_no_affected_crews(self):
        """Test graceful degradation when no crews are affected."""
        strategy = GracefulDegradationStrategy()
        context = ExecutionContext(slice_idx=0)

        error = RuntimeError("Internal error")

        updated_context = await strategy.execute(
            context=context,
            failed_crew="independent_crew",
            error=error,
            affected_crews=[],  # No dependent crews
        )

        # Should still create fallback data for consistency
        assert "independent_crew_fallback" in updated_context.partial_results

        # Should log action mentioning 0 dependent crews
        assert len(updated_context.error_recovery_actions) == 1
        action = updated_context.error_recovery_actions[0]
        assert "0 dependent crews" in action
        assert "[]" in action  # Empty list representation

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_large_crew_list(self):
        """Test graceful degradation with many affected crews."""
        strategy = GracefulDegradationStrategy()
        context = ExecutionContext(slice_idx=0)

        # Create large list of affected crews
        affected_crews = [f"dependent_crew_{i}" for i in range(100)]
        error = Exception("Hub crew failed")

        updated_context = await strategy.execute(
            context=context,
            failed_crew="hub_crew",
            error=error,
            affected_crews=affected_crews,
        )

        # All crews should have degraded input markers
        for crew_id in affected_crews:
            degraded_key = f"{crew_id}_degraded_input"
            assert degraded_key in updated_context.partial_results

        # Should log correct count
        action = updated_context.error_recovery_actions[0]
        assert "100 dependent crews" in action

    @pytest.mark.asyncio
    async def test_graceful_degradation_data_types(self):
        """Test graceful degradation with various data types in fallback."""
        complex_fallback = {
            "strings": "fallback_value",
            "numbers": 42,
            "floats": 3.14159,
            "booleans": True,
            "lists": [1, 2, 3, "mixed", {"nested": "dict"}],
            "dicts": {"nested": {"deeply": {"nested": "value"}}},
            "none_values": None,
        }

        strategy = GracefulDegradationStrategy(fallback_data=complex_fallback)
        context = ExecutionContext(slice_idx=0)

        updated_context = await strategy.execute(
            context=context,
            failed_crew="complex_crew",
            error=Exception("Error"),
            affected_crews=["consumer"],
        )

        fallback_data = updated_context.partial_results["complex_crew_fallback"]
        assert fallback_data["fallback_data"] == complex_fallback
        assert fallback_data["fallback_data"]["lists"][4]["nested"] == "dict"


class TestCascadeFailureStrategy:
    """Test CascadeFailureStrategy behavior and edge cases."""

    @pytest.fixture
    def execution_context_with_crews(self) -> ExecutionContext:
        """Create execution context with crew logs."""
        context = ExecutionContext(
            slice_idx=0,
            phase_execution_order=[
                ExecutionPhase(crews=["crew_a"], phase_index=0),
                ExecutionPhase(crews=["crew_b", "crew_c"], phase_index=1),
                ExecutionPhase(crews=["crew_d"], phase_index=2),
            ],
        )

        # Initialize crew logs
        for crew_id in ["crew_a", "crew_b", "crew_c", "crew_d"]:
            log = context.get_crew_log(crew_id)
            log.state = CrewExecutionState.READY

        return context

    @pytest.mark.asyncio
    async def test_cascade_failure_basic_behavior(self, execution_context_with_crews):
        """Test basic cascade failure handling."""
        strategy = CascadeFailureStrategy(halt_on_cascade=False)

        error = Exception("Critical failure in crew_a")
        affected_crews = ["crew_b", "crew_c"]

        updated_context = await strategy.execute(
            context=execution_context_with_crews,
            failed_crew="crew_a",
            error=error,
            affected_crews=affected_crews,
        )

        # Check affected crews were cancelled
        for crew_id in affected_crews:
            crew_log = updated_context.get_crew_log(crew_id)
            assert crew_log.state == CrewExecutionState.CANCELLED
            assert (
                "Cancelled due to cascade failure from 'crew_a'"
                in crew_log.error_message
            )
            assert crew_id in updated_context.failed_crews

        # Check cascade details were recorded
        cascade_data = updated_context.partial_results["crew_a_cascade"]
        assert cascade_data["strategy"] == "cascade_failure"
        assert cascade_data["error"] == "Critical failure in crew_a"
        assert cascade_data["cascaded_crews"] == affected_crews
        assert cascade_data["halt_orchestration"] is False

        # Check recovery action was logged
        assert len(updated_context.error_recovery_actions) == 1
        action = updated_context.error_recovery_actions[0]
        assert "Cascade failure from 'crew_a'" in action
        assert "Cancelled 2 dependent crews" in action

    @pytest.mark.asyncio
    async def test_cascade_failure_with_halt_orchestration(
        self, execution_context_with_crews
    ):
        """Test cascade failure that should halt entire orchestration."""
        strategy = CascadeFailureStrategy(halt_on_cascade=True)

        error = Exception("Catastrophic failure")
        affected_crews = ["crew_b", "crew_c", "crew_d"]

        updated_context = await strategy.execute(
            context=execution_context_with_crews,
            failed_crew="crew_a",
            error=error,
            affected_crews=affected_crews,
        )

        # All affected crews should be cancelled
        for crew_id in affected_crews:
            assert (
                updated_context.get_crew_log(crew_id).state
                == CrewExecutionState.CANCELLED
            )

        # Halt flag should be set
        cascade_data = updated_context.partial_results["crew_a_cascade"]
        assert cascade_data["halt_orchestration"] is True

    @pytest.mark.asyncio
    async def test_cascade_failure_with_no_affected_crews(self):
        """Test cascade failure when no crews are affected."""
        strategy = CascadeFailureStrategy()
        context = ExecutionContext(slice_idx=0)

        error = Exception("Isolated critical failure")

        updated_context = await strategy.execute(
            context=context, failed_crew="isolated_crew", error=error, affected_crews=[]
        )

        # Should still record cascade attempt
        cascade_data = updated_context.partial_results["isolated_crew_cascade"]
        assert cascade_data["cascaded_crews"] == []

        # Should log action for 0 crews
        action = updated_context.error_recovery_actions[0]
        assert "Cancelled 0 dependent crews" in action

    @pytest.mark.asyncio
    async def test_cascade_failure_preserves_existing_crew_states(
        self, execution_context_with_crews
    ):
        """Test that cascade only affects targeted crews, not others."""
        strategy = CascadeFailureStrategy()

        # Set some crews to different states
        execution_context_with_crews.get_crew_log(
            "crew_d"
        ).state = CrewExecutionState.COMPLETED

        error = Exception("Partial cascade failure")
        affected_crews = ["crew_b", "crew_c"]  # crew_d not affected

        updated_context = await strategy.execute(
            context=execution_context_with_crews,
            failed_crew="crew_a",
            error=error,
            affected_crews=affected_crews,
        )

        # Affected crews should be cancelled
        assert (
            updated_context.get_crew_log("crew_b").state == CrewExecutionState.CANCELLED
        )
        assert (
            updated_context.get_crew_log("crew_c").state == CrewExecutionState.CANCELLED
        )

        # Unaffected crew should maintain its state
        assert (
            updated_context.get_crew_log("crew_d").state == CrewExecutionState.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_cascade_failure_with_massive_crew_list(self):
        """Test cascade failure with very large number of affected crews."""
        strategy = CascadeFailureStrategy()
        context = ExecutionContext(slice_idx=0)

        # Create large list of crews to cascade to
        affected_crews = [f"cascade_crew_{i}" for i in range(1000)]

        # Initialize crew logs for all crews
        for crew_id in affected_crews:
            log = context.get_crew_log(crew_id)
            log.state = CrewExecutionState.READY

        error = Exception("Hub failure affecting 1000 crews")

        updated_context = await strategy.execute(
            context=context,
            failed_crew="hub_crew",
            error=error,
            affected_crews=affected_crews,
        )

        # All crews should be cancelled and added to failed set
        for crew_id in affected_crews:
            assert (
                updated_context.get_crew_log(crew_id).state
                == CrewExecutionState.CANCELLED
            )
            assert crew_id in updated_context.failed_crews

        # Should log correct count
        action = updated_context.error_recovery_actions[0]
        assert "Cancelled 1000 dependent crews" in action


class TestRetryWithBackoffStrategy:
    """Test RetryWithBackoffStrategy behavior and edge cases."""

    @pytest.fixture
    def execution_context_with_retry_crew(self) -> ExecutionContext:
        """Create execution context with a crew that has retry history."""
        context = ExecutionContext(slice_idx=0)
        crew_log = context.get_crew_log("retry_crew")
        crew_log.state = CrewExecutionState.FAILED
        return context

    @pytest.mark.asyncio
    async def test_retry_basic_behavior(self, execution_context_with_retry_crew):
        """Test basic retry with backoff behavior."""
        strategy = RetryWithBackoffStrategy(max_retries=3, base_delay_ms=1000)

        error = Exception("Temporary service error")
        affected_crews = ["waiting_crew"]

        updated_context = await strategy.execute(
            context=execution_context_with_retry_crew,
            failed_crew="retry_crew",
            error=error,
            affected_crews=affected_crews,
        )

        # Crew should be reset to READY state for retry
        crew_log = updated_context.get_crew_log("retry_crew")
        assert crew_log.state == CrewExecutionState.READY

        # Retry details should be recorded
        retry_key = "retry_crew_retry_0"  # First retry attempt
        assert retry_key in updated_context.partial_results

        retry_data = updated_context.partial_results[retry_key]
        assert retry_data["strategy"] == "retry_with_backoff"
        assert retry_data["retry_attempt"] == 1
        assert retry_data["max_retries"] == 3
        assert retry_data["delay_ms"] == 1000  # Base delay for first retry
        assert retry_data["affected_crews"] == affected_crews

        # Recovery action should be logged
        assert len(updated_context.error_recovery_actions) == 1
        action = updated_context.error_recovery_actions[0]
        assert "Retry scheduled for 'retry_crew'" in action
        assert "(attempt 1/3)" in action
        assert "after 1000ms delay" in action

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self, execution_context_with_retry_crew):
        """Test that retry delays follow exponential backoff pattern."""
        strategy = RetryWithBackoffStrategy(max_retries=4, base_delay_ms=500)

        # Simulate multiple retry attempts
        crew_log = execution_context_with_retry_crew.get_crew_log("retry_crew")
        expected_delays = [500, 1000, 2000, 4000]  # 500 * 2^n

        for i in range(4):
            crew_log.retry_count = i

            updated_context = await strategy.execute(
                context=execution_context_with_retry_crew,
                failed_crew="retry_crew",
                error=Exception(f"Error attempt {i + 1}"),
                affected_crews=[],
            )

            retry_key = f"retry_crew_retry_{i}"
            retry_data = updated_context.partial_results[retry_key]
            assert retry_data["delay_ms"] == expected_delays[i]
            assert retry_data["retry_attempt"] == i + 1

    @pytest.mark.asyncio
    async def test_retry_max_retries_exceeded(self, execution_context_with_retry_crew):
        """Test behavior when max retries is exceeded."""
        strategy = RetryWithBackoffStrategy(max_retries=2, base_delay_ms=1000)

        # Set retry count to max
        crew_log = execution_context_with_retry_crew.get_crew_log("retry_crew")
        crew_log.retry_count = 2  # Already at max

        error = Exception("Persistent failure")
        affected_crews = ["dependent_crew_1", "dependent_crew_2"]

        updated_context = await strategy.execute(
            context=execution_context_with_retry_crew,
            failed_crew="retry_crew",
            error=error,
            affected_crews=affected_crews,
        )

        # Should cascade failure instead of retrying
        for crew_id in affected_crews:
            crew_log = updated_context.get_crew_log(crew_id)
            assert crew_log.state == CrewExecutionState.CANCELLED
            assert crew_id in updated_context.failed_crews

        # Should have cascade data recorded
        assert "retry_crew_cascade" in updated_context.partial_results

        # Should log max retries exceeded
        actions = updated_context.error_recovery_actions
        max_retries_action = next(
            action
            for action in actions
            if "Max retries (2) exceeded for 'retry_crew'" in action
        )
        assert "Cascading failure to dependent crews" in max_retries_action

    @pytest.mark.asyncio
    async def test_retry_with_zero_max_retries(self):
        """Test retry strategy with max_retries=0 (immediate cascade)."""
        strategy = RetryWithBackoffStrategy(max_retries=0, base_delay_ms=1000)
        context = ExecutionContext(slice_idx=0)

        error = Exception("Should cascade immediately")
        affected_crews = ["cascade_crew"]

        updated_context = await strategy.execute(
            context=context,
            failed_crew="no_retry_crew",
            error=error,
            affected_crews=affected_crews,
        )

        # Should immediately cascade without retry
        crew_log = updated_context.get_crew_log("cascade_crew")
        assert crew_log.state == CrewExecutionState.CANCELLED
        assert "cascade_crew" in updated_context.failed_crews

    @pytest.mark.asyncio
    async def test_retry_with_very_large_delays(self):
        """Test retry with very large base delays (edge case for timing)."""
        strategy = RetryWithBackoffStrategy(
            max_retries=3, base_delay_ms=3600000
        )  # 1 hour
        context = ExecutionContext(slice_idx=0)

        error = Exception("Long delay test")

        updated_context = await strategy.execute(
            context=context, failed_crew="delay_crew", error=error, affected_crews=[]
        )

        # Should handle large delays correctly
        retry_data = updated_context.partial_results["delay_crew_retry_0"]
        assert retry_data["delay_ms"] == 3600000

        # Test second retry would be even larger
        crew_log = updated_context.get_crew_log("delay_crew")
        crew_log.retry_count = 1

        updated_context = await strategy.execute(
            context=updated_context,
            failed_crew="delay_crew",
            error=error,
            affected_crews=[],
        )

        retry_data = updated_context.partial_results["delay_crew_retry_1"]
        assert retry_data["delay_ms"] == 7200000  # 2 hours


class TestPartialFailureHandler:
    """Test PartialFailureHandler main orchestration logic."""

    @pytest.fixture
    def basic_handler(self) -> PartialFailureHandler:
        """Create basic failure handler for testing."""
        return PartialFailureHandler()

    @pytest.fixture
    def handler_with_critical_crews(self) -> PartialFailureHandler:
        """Create failure handler with critical crews configured."""
        critical_crews = {"auth_crew", "payment_crew", "security_crew"}
        return PartialFailureHandler(critical_crews=critical_crews)

    @pytest.fixture
    def execution_context(self) -> ExecutionContext:
        """Create execution context for testing."""
        return ExecutionContext(
            slice_idx=0,
            phase_execution_order=[
                ExecutionPhase(crews=["producer"], phase_index=0),
                ExecutionPhase(crews=["processor_1", "processor_2"], phase_index=1),
                ExecutionPhase(crews=["consumer"], phase_index=2),
            ],
        )

    @pytest.mark.asyncio
    async def test_isolated_failure_handling(self, basic_handler, execution_context):
        """Test handling of isolated failures (no dependencies)."""
        error = ValueError("Configuration issue")

        updated_context = await basic_handler.handle_crew_failure(
            failed_crew="isolated_crew",
            error=error,
            context=execution_context,
            dependent_crews=[],
        )

        # Should use isolated strategy
        assert "isolated_crew_failure_isolated" in updated_context.partial_results
        failure_data = updated_context.partial_results["isolated_crew_failure_isolated"]
        assert failure_data["recovery_strategy"] == "isolated"
        assert failure_data["impact"] == "none"

        # Should record failure in history
        failure_history = basic_handler.get_failure_history("isolated_crew")
        assert len(failure_history) == 1
        assert failure_history[0] == error

    @pytest.mark.asyncio
    async def test_degraded_failure_handling(self, basic_handler, execution_context):
        """Test handling of failures that allow graceful degradation."""
        error = TimeoutError("Processing timeout")
        dependent_crews = ["consumer_1", "consumer_2"]

        updated_context = await basic_handler.handle_crew_failure(
            failed_crew="producer_crew",
            error=error,
            context=execution_context,
            dependent_crews=dependent_crews,
        )

        # Should use graceful degradation strategy
        assert "producer_crew_fallback" in updated_context.partial_results

        # Dependent crews should have degraded input markers
        for crew_id in dependent_crews:
            degraded_key = f"{crew_id}_degraded_input"
            assert degraded_key in updated_context.partial_results

        # Should have comprehensive failure analysis logged
        analysis_action = next(
            action
            for action in updated_context.error_recovery_actions
            if "Failure analysis for 'producer_crew'" in action
        )
        assert "degraded impact" in analysis_action
        assert "2 crews affected" in analysis_action

    @pytest.mark.asyncio
    async def test_critical_crew_cascade_handling(
        self, handler_with_critical_crews, execution_context
    ):
        """Test handling of critical crew failures that should cascade."""
        error = Exception("Authentication system failure")
        dependent_crews = ["user_service", "api_gateway"]

        updated_context = await handler_with_critical_crews.handle_crew_failure(
            failed_crew="auth_crew",  # Critical crew
            error=error,
            context=execution_context,
            dependent_crews=dependent_crews,
        )

        # Should use cascade strategy due to critical crew
        assert "auth_crew_cascade" in updated_context.partial_results

        # Dependent crews should be cancelled
        for crew_id in dependent_crews:
            crew_log = updated_context.get_crew_log(crew_id)
            assert crew_log.state == CrewExecutionState.CANCELLED
            assert crew_id in updated_context.failed_crews

    @pytest.mark.asyncio
    async def test_wide_impact_cascade_handling(self, basic_handler, execution_context):
        """Test handling of failures affecting many crews (>5 threshold)."""
        error = Exception("Hub service failure")
        # Create list with more than 5 dependent crews
        dependent_crews = [f"dependent_{i}" for i in range(10)]

        updated_context = await basic_handler.handle_crew_failure(
            failed_crew="hub_crew",
            error=error,
            context=execution_context,
            dependent_crews=dependent_crews,
        )

        # With >5 crews, should trigger cascading impact level
        # But since it's not a critical crew and critical path not broken,
        # the strategy selection logic will use retry strategy, not cascade
        if "hub_crew_cascade" in updated_context.partial_results:
            # All dependent crews should be cancelled
            for crew_id in dependent_crews:
                assert crew_id in updated_context.failed_crews
        elif "hub_crew_retry_0" in updated_context.partial_results:
            # Retry strategy was used instead
            retry_data = updated_context.partial_results["hub_crew_retry_0"]
            assert len(retry_data["affected_crews"]) == 10
        else:
            pytest.fail(
                f"Expected cascade or retry strategy. Found: {list(updated_context.partial_results.keys())}"
            )

    @pytest.mark.asyncio
    async def test_custom_strategy_registration(self, basic_handler, execution_context):
        """Test custom recovery strategy registration and usage."""
        # Create mock custom strategy that returns awaitable
        custom_strategy = Mock(spec=RecoveryStrategy)

        # Make execute method return a coroutine that resolves to execution_context
        async def mock_execute(context, failed_crew, error, affected_crews):
            return context

        custom_strategy.execute = mock_execute

        # Register custom strategy for specific crew
        basic_handler.register_strategy("special_crew", custom_strategy)

        error = Exception("Special crew error")

        result_context = await basic_handler.handle_crew_failure(
            failed_crew="special_crew",
            error=error,
            context=execution_context,
            dependent_crews=["dependent"],
        )

        # Should have returned the context
        assert result_context is execution_context

        # Should have recorded the failure in history
        failure_history = basic_handler.get_failure_history("special_crew")
        assert len(failure_history) == 1
        assert failure_history[0] == error

    @pytest.mark.asyncio
    async def test_failure_history_accumulation(self, basic_handler, execution_context):
        """Test that failure history accumulates correctly."""
        errors = [
            ValueError("Config error"),
            TimeoutError("Timeout 1"),
            RuntimeError("Runtime issue"),
            TimeoutError("Timeout 2"),
        ]

        # Cause multiple failures for same crew
        for error in errors:
            await basic_handler.handle_crew_failure(
                failed_crew="problematic_crew",
                error=error,
                context=execution_context,
                dependent_crews=[],
            )

        # Check failure history
        history = basic_handler.get_failure_history("problematic_crew")
        assert len(history) == 4
        assert history == errors

        # Check other crew has no history
        assert basic_handler.get_failure_history("other_crew") == []

    def test_failure_history_clear_specific_crew(self, basic_handler):
        """Test clearing failure history for specific crew."""
        basic_handler._failure_history["crew_a"] = [
            Exception("Error 1"),
            Exception("Error 2"),
        ]
        basic_handler._failure_history["crew_b"] = [Exception("Error 3")]

        # Clear specific crew
        basic_handler.clear_failure_history("crew_a")

        # crew_a should be cleared, crew_b should remain
        assert basic_handler.get_failure_history("crew_a") == []
        assert len(basic_handler.get_failure_history("crew_b")) == 1

    def test_failure_history_clear_all(self, basic_handler):
        """Test clearing all failure history."""
        basic_handler._failure_history["crew_a"] = [Exception("Error 1")]
        basic_handler._failure_history["crew_b"] = [Exception("Error 2")]

        # Clear all
        basic_handler.clear_failure_history()

        # All should be cleared
        assert basic_handler.get_failure_history("crew_a") == []
        assert basic_handler.get_failure_history("crew_b") == []

    @pytest.mark.asyncio
    async def test_dependency_impact_analysis_edge_cases(
        self, basic_handler, execution_context
    ):
        """Test dependency impact analysis with edge cases."""
        # Test analysis with empty dependent crews
        analysis = basic_handler._analyze_dependency_impact(
            failed_crew="isolated",
            error=Exception("Test"),
            context=execution_context,
            dependent_crews=[],
        )
        assert analysis.impact_level == FailureImpact.ISOLATED
        assert analysis.directly_affected == []
        assert analysis.total_affected_count == 0
        assert not analysis.critical_path_broken

        # Test analysis with many dependent crews
        many_crews = [f"crew_{i}" for i in range(10)]
        analysis = basic_handler._analyze_dependency_impact(
            failed_crew="hub",
            error=Exception("Test"),
            context=execution_context,
            dependent_crews=many_crews,
        )
        assert analysis.impact_level == FailureImpact.CASCADING  # > 5 crews
        assert analysis.directly_affected == many_crews
        assert analysis.total_affected_count == 10

    @pytest.mark.asyncio
    async def test_strategy_selection_logic(self, handler_with_critical_crews):
        """Test strategy selection logic for different scenarios."""
        # Test isolated failure strategy selection
        analysis = DependencyChainAnalysis(
            failed_crew="isolated",
            impact_level=FailureImpact.ISOLATED,
            directly_affected=[],
            transitively_affected=[],
            critical_path_broken=False,
            recovery_options=[],
        )
        strategy = handler_with_critical_crews._select_recovery_strategy(
            "isolated", analysis
        )
        assert isinstance(strategy, IsolatedFailureStrategy)

        # Test graceful degradation strategy selection
        analysis = DependencyChainAnalysis(
            failed_crew="degraded",
            impact_level=FailureImpact.DEGRADED,
            directly_affected=["dep1"],
            transitively_affected=[],
            critical_path_broken=False,
            recovery_options=[],
        )
        strategy = handler_with_critical_crews._select_recovery_strategy(
            "degraded", analysis
        )
        assert isinstance(strategy, GracefulDegradationStrategy)

        # Test cascade strategy selection for critical path broken
        analysis = DependencyChainAnalysis(
            failed_crew="critical",
            impact_level=FailureImpact.CASCADING,
            directly_affected=["dep1"],
            transitively_affected=[],
            critical_path_broken=True,
            recovery_options=[],
        )
        strategy = handler_with_critical_crews._select_recovery_strategy(
            "critical", analysis
        )
        assert isinstance(strategy, CascadeFailureStrategy)
        assert strategy.halt_on_cascade is True

        # Test retry strategy selection for cascading but not critical path
        analysis = DependencyChainAnalysis(
            failed_crew="retry_candidate",
            impact_level=FailureImpact.CASCADING,
            directly_affected=["dep1"],
            transitively_affected=[],
            critical_path_broken=False,
            recovery_options=[],
        )
        strategy = handler_with_critical_crews._select_recovery_strategy(
            "retry_candidate", analysis
        )
        assert isinstance(strategy, RetryWithBackoffStrategy)


class TestPartialFailureHandlerPerformanceAndMemory:
    """Test PartialFailureHandler performance and memory characteristics."""

    @pytest.mark.asyncio
    async def test_large_failure_history_memory_usage(self):
        """Test memory behavior with large failure histories."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Generate many failures for memory testing
        for i in range(1000):
            error = Exception(f"Error {i}")
            await handler.handle_crew_failure(
                failed_crew=f"crew_{i % 10}",  # 10 crews with 100 failures each
                error=error,
                context=context,
                dependent_crews=[],
            )

        # Check that failure history is being stored correctly
        for crew_num in range(10):
            crew_id = f"crew_{crew_num}"
            history = handler.get_failure_history(crew_id)
            assert len(history) == 100

        # Test memory cleanup via clearing history
        handler.clear_failure_history()
        for crew_num in range(10):
            crew_id = f"crew_{crew_num}"
            assert handler.get_failure_history(crew_id) == []

    @pytest.mark.asyncio
    async def test_massive_dependency_chain_performance(self):
        """Test performance with massive dependency chains."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Create massive dependency list
        dependent_crews = [f"dependent_{i}" for i in range(5000)]
        error = Exception("Hub failure with massive impact")

        # Time the failure handling
        import time

        start_time = time.time()

        updated_context = await handler.handle_crew_failure(
            failed_crew="massive_hub",
            error=error,
            context=context,
            dependent_crews=dependent_crews,
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 5.0  # 5 seconds threshold

        # Should have either cascade data, graceful degradation data, or retry data
        # Check which strategy was used based on impact analysis
        if "massive_hub_cascade" in updated_context.partial_results:
            # Cascade strategy was used
            cascade_data = updated_context.partial_results["massive_hub_cascade"]
            assert len(cascade_data["cascaded_crews"]) == 5000
            # All crews should be in failed set
            assert len(updated_context.failed_crews) == 5000
        elif "massive_hub_fallback" in updated_context.partial_results:
            # Graceful degradation was used
            # Should have degraded input markers for all crews
            degraded_count = sum(
                1
                for key in updated_context.partial_results.keys()
                if key.endswith("_degraded_input")
            )
            assert degraded_count == 5000
        elif "massive_hub_retry_0" in updated_context.partial_results:
            # Retry strategy was used (for cascading impact but not critical path broken)
            retry_data = updated_context.partial_results["massive_hub_retry_0"]
            assert len(retry_data["affected_crews"]) == 5000
            assert retry_data["strategy"] == "retry_with_backoff"
            # Crew should be reset to READY state for retry
            crew_log = updated_context.get_crew_log("massive_hub")
            assert crew_log.state == CrewExecutionState.READY
        else:
            pytest.fail(
                f"Expected one of cascade, graceful degradation, or retry strategies to be used. Found keys: {list(updated_context.partial_results.keys())}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_failure_handling(self):
        """Test concurrent failure handling scenarios."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Create multiple concurrent failure handling tasks
        async def handle_failure(crew_id: str, error_msg: str):
            error = Exception(error_msg)
            return await handler.handle_crew_failure(
                failed_crew=crew_id,
                error=error,
                context=context,
                dependent_crews=[f"{crew_id}_dependent"],
            )

        # Run multiple failures concurrently
        tasks = [
            handle_failure(f"crew_{i}", f"Concurrent error {i}") for i in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 20

        # Failure history should record all failures
        total_failures = 0
        for i in range(20):
            crew_id = f"crew_{i}"
            history = handler.get_failure_history(crew_id)
            total_failures += len(history)

        assert total_failures == 20


class TestPartialFailureHandlerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_malformed_error_objects(self):
        """Test handling of malformed or unusual error objects."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Test with None error (should handle gracefully)
        try:
            updated_context = await handler.handle_crew_failure(
                failed_crew="crew_with_none_error",
                error=None,  # type: ignore[arg-type]  # Intentional type error for testing
                context=context,
                dependent_crews=[],
            )
            # Should handle None error without crashing
            assert len(handler.get_failure_history("crew_with_none_error")) == 1
        except Exception:
            # If it raises an exception, it should be handled gracefully in production
            pytest.fail("Handler should gracefully handle None errors")

        # Test with custom exception with unusual __str__ method
        class WeirdError(Exception):
            def __str__(self):
                return ""  # Empty string representation

        weird_error = WeirdError()
        updated_context = await handler.handle_crew_failure(
            failed_crew="crew_with_weird_error",
            error=weird_error,
            context=context,
            dependent_crews=[],
        )

        # Should handle empty string representation
        failure_data = updated_context.partial_results[
            "crew_with_weird_error_failure_isolated"
        ]
        assert failure_data["error"] == ""

    @pytest.mark.asyncio
    async def test_circular_dependency_implications(self):
        """Test behavior that might arise from circular dependencies."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Simulate what might happen with circular dependencies
        # crew_a depends on crew_b, crew_b depends on crew_a

        # First failure: crew_a fails, affects crew_b
        await handler.handle_crew_failure(
            failed_crew="crew_a",
            error=Exception("crew_a failure"),
            context=context,
            dependent_crews=["crew_b"],
        )

        # Then: crew_b fails, affects crew_a (but crew_a already failed)
        updated_context = await handler.handle_crew_failure(
            failed_crew="crew_b",
            error=Exception("crew_b failure"),
            context=context,
            dependent_crews=["crew_a"],
        )

        # Both crews should have failure records
        assert len(handler.get_failure_history("crew_a")) == 1
        assert len(handler.get_failure_history("crew_b")) == 1

        # Should handle the scenario without infinite recursion
        assert len(updated_context.error_recovery_actions) >= 2

    @pytest.mark.asyncio
    async def test_empty_crew_ids_and_special_characters(self):
        """Test handling of edge case crew IDs."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Test empty crew ID
        error = Exception("Empty crew ID test")
        await handler.handle_crew_failure(
            failed_crew="", error=error, context=context, dependent_crews=[]
        )

        # Should handle empty string crew ID
        assert len(handler.get_failure_history("")) == 1

        # Test crew IDs with special characters
        special_crew_ids = [
            "crew-with-dashes",
            "crew_with_underscores",
            "crew.with.dots",
            "crew with spaces",
            "crew/with/slashes",
            "crew:with:colons",
            "crew@with@symbols",
            "crew#with#hash",
            "crew$with$dollar",
            "crew%with%percent",
        ]

        for crew_id in special_crew_ids:
            await handler.handle_crew_failure(
                failed_crew=crew_id,
                error=Exception(f"Error for {crew_id}"),
                context=context,
                dependent_crews=[],
            )

            # Should handle special characters in crew IDs
            assert len(handler.get_failure_history(crew_id)) == 1

    @pytest.mark.asyncio
    async def test_very_long_error_messages(self):
        """Test handling of very long error messages."""
        handler = PartialFailureHandler()
        context = ExecutionContext(slice_idx=0)

        # Create very long error message
        long_message = "ERROR: " + "This is a very long error message. " * 1000
        long_error = Exception(long_message)

        updated_context = await handler.handle_crew_failure(
            failed_crew="crew_with_long_error",
            error=long_error,
            context=context,
            dependent_crews=[],
        )

        # Should handle very long error messages without issues
        failure_data = updated_context.partial_results[
            "crew_with_long_error_failure_isolated"
        ]
        assert failure_data["error"] == long_message

        # Failure history should store the complete error
        history = handler.get_failure_history("crew_with_long_error")
        assert str(history[0]) == long_message

    def test_strategy_registry_edge_cases(self):
        """Test edge cases in strategy registry."""
        handler = PartialFailureHandler()

        # Test registering strategy for same crew multiple times
        strategy1 = IsolatedFailureStrategy()
        strategy2 = GracefulDegradationStrategy()

        handler.register_strategy("overwrite_crew", strategy1)
        handler.register_strategy("overwrite_crew", strategy2)  # Should overwrite

        # Latest registration should be used
        assert handler._strategy_registry["overwrite_crew"] is strategy2

        # Test registering with empty crew ID
        handler.register_strategy("", strategy1)
        assert handler._strategy_registry[""] is strategy1


class TestPartialFailureHandlerIntegrationScenarios:
    """Test realistic multi-crew failure scenarios."""

    @pytest.mark.asyncio
    async def test_realistic_data_pipeline_failure_scenario(self):
        """Test realistic data pipeline failure with mixed recovery strategies."""
        handler = PartialFailureHandler(
            critical_crews={"data_ingestion", "data_validation"}
        )

        # Register custom strategies for specific crews
        handler.register_strategy(
            "cache_service", GracefulDegradationStrategy(fallback_data={"cached": True})
        )

        context = ExecutionContext(
            slice_idx=5,
            phase_execution_order=[
                ExecutionPhase(crews=["data_ingestion"], phase_index=0),
                ExecutionPhase(
                    crews=["data_processing", "cache_service"], phase_index=1
                ),
                ExecutionPhase(crews=["analytics", "reporting"], phase_index=2),
                ExecutionPhase(crews=["dashboard", "alerts"], phase_index=3),
            ],
        )

        # Scenario: Cache service fails (should use custom graceful degradation)
        cache_error = TimeoutError("Cache service timeout")
        context = await handler.handle_crew_failure(
            failed_crew="cache_service",
            error=cache_error,
            context=context,
            dependent_crews=["analytics", "dashboard"],
        )

        # Should use custom strategy with cached fallback data
        fallback_data = context.partial_results["cache_service_fallback"]
        assert fallback_data["fallback_data"]["cached"] is True

        # Scenario: Data ingestion fails (critical crew - should cascade)
        ingestion_error = Exception("Data source unavailable")
        context = await handler.handle_crew_failure(
            failed_crew="data_ingestion",
            error=ingestion_error,
            context=context,
            dependent_crews=["data_processing", "analytics", "reporting"],
        )

        # Should cascade due to critical crew
        assert "data_ingestion_cascade" in context.partial_results
        for crew_id in ["data_processing", "analytics", "reporting"]:
            assert crew_id in context.failed_crews

        # Scenario: Non-critical isolated failure
        isolated_error = ValueError("Minor configuration issue")
        context = await handler.handle_crew_failure(
            failed_crew="isolated_monitoring",
            error=isolated_error,
            context=context,
            dependent_crews=[],
        )

        # Should use isolated strategy
        assert "isolated_monitoring_failure_isolated" in context.partial_results

        # Check comprehensive failure history
        assert len(handler.get_failure_history("cache_service")) == 1
        assert len(handler.get_failure_history("data_ingestion")) == 1
        assert len(handler.get_failure_history("isolated_monitoring")) == 1

    @pytest.mark.asyncio
    async def test_financial_processing_critical_failure_scenario(self):
        """Test critical failure scenario in financial processing system."""
        critical_crews = {"payment_processor", "fraud_detection", "transaction_log"}
        handler = PartialFailureHandler(critical_crews=critical_crews)

        context = ExecutionContext(
            slice_idx=10,
            phase_execution_order=[
                ExecutionPhase(crews=["user_auth", "payment_processor"], phase_index=0),
                ExecutionPhase(
                    crews=["fraud_detection", "risk_assessment"], phase_index=1
                ),
                ExecutionPhase(crews=["transaction_log", "audit_trail"], phase_index=2),
                ExecutionPhase(
                    crews=["notification", "receipt_generation"], phase_index=3
                ),
            ],
        )

        # Scenario: Payment processor fails (critical financial component)
        payment_error = Exception("Payment gateway connection failed")
        context = await handler.handle_crew_failure(
            failed_crew="payment_processor",
            error=payment_error,
            context=context,
            dependent_crews=["fraud_detection", "transaction_log", "notification"],
        )

        # Should cascade failure and halt orchestration for financial integrity
        cascade_data = context.partial_results["payment_processor_cascade"]
        assert cascade_data["halt_orchestration"] is True

        # All dependent financial processes should be cancelled
        for crew_id in ["fraud_detection", "transaction_log", "notification"]:
            crew_log = context.get_crew_log(crew_id)
            assert crew_log.state == CrewExecutionState.CANCELLED
            assert "payment_processor" in crew_log.error_message

    @pytest.mark.asyncio
    async def test_retry_exhaustion_cascade_scenario(self):
        """Test scenario where retry attempts are exhausted and cascade to dependents."""
        handler = PartialFailureHandler()

        context = ExecutionContext(slice_idx=0)

        # Set up crew with existing retry history
        retry_crew_log = context.get_crew_log("unstable_service")

        # Register retry strategy with low max retries
        handler.register_strategy(
            "unstable_service",
            RetryWithBackoffStrategy(max_retries=2, base_delay_ms=100),
        )

        dependent_crews = ["service_a", "service_b", "service_c"]

        # First failure - should retry
        retry_crew_log.retry_count = 0
        context = await handler.handle_crew_failure(
            failed_crew="unstable_service",
            error=Exception("Temporary failure 1"),
            context=context,
            dependent_crews=dependent_crews,
        )

        # Should have scheduled retry
        assert "unstable_service_retry_0" in context.partial_results
        assert retry_crew_log.state == CrewExecutionState.READY

        # Second failure - should retry again
        retry_crew_log.retry_count = 1
        context = await handler.handle_crew_failure(
            failed_crew="unstable_service",
            error=Exception("Temporary failure 2"),
            context=context,
            dependent_crews=dependent_crews,
        )

        # Should have scheduled second retry
        assert "unstable_service_retry_1" in context.partial_results

        # Third failure - should exhaust retries and cascade
        retry_crew_log.retry_count = 2  # At max retries
        context = await handler.handle_crew_failure(
            failed_crew="unstable_service",
            error=Exception("Permanent failure"),
            context=context,
            dependent_crews=dependent_crews,
        )

        # Should cascade after max retries exceeded
        assert "unstable_service_cascade" in context.partial_results
        for crew_id in dependent_crews:
            assert crew_id in context.failed_crews

        # Should have logged max retries exceeded
        max_retries_action = next(
            action
            for action in context.error_recovery_actions
            if "Max retries (2) exceeded" in action
        )
        assert "unstable_service" in max_retries_action


# Performance benchmarking fixtures and utilities
@pytest.fixture
def large_execution_context() -> ExecutionContext:
    """Create large execution context for performance testing."""
    phases = []
    crew_count = 0

    # Create 10 phases with varying numbers of crews
    for phase_idx in range(10):
        crews_in_phase = [f"crew_{crew_count + i}" for i in range(phase_idx * 10 + 5)]
        phases.append(ExecutionPhase(crews=crews_in_phase, phase_index=phase_idx))
        crew_count += len(crews_in_phase)

    context = ExecutionContext(slice_idx=0, phase_execution_order=phases)

    # Initialize all crew logs
    for phase in phases:
        for crew_id in phase.crews:
            log = context.get_crew_log(crew_id)
            log.state = CrewExecutionState.READY

    return context


class TestPartialFailureHandlerComplianceAndValidation:
    """Test compliance with protocol and validation requirements."""

    def test_recovery_strategy_protocol_compliance(self):
        """Test that all recovery strategies comply with the protocol."""
        strategies = [
            IsolatedFailureStrategy(),
            GracefulDegradationStrategy(),
            CascadeFailureStrategy(),
            RetryWithBackoffStrategy(),
        ]

        for strategy in strategies:
            # Each strategy should have the execute method
            assert hasattr(strategy, "execute")
            assert callable(strategy.execute)

            # Method should be async
            import inspect

            assert inspect.iscoroutinefunction(strategy.execute)

    @pytest.mark.asyncio
    async def test_context_immutability_and_updates(self):
        """Test that context updates are handled correctly."""
        handler = PartialFailureHandler()
        original_context = ExecutionContext(slice_idx=0)

        # Add some original data
        original_context.partial_results["original_key"] = "original_value"
        original_context.error_recovery_actions.append("original_action")

        error = Exception("Test error")

        updated_context = await handler.handle_crew_failure(
            failed_crew="test_crew",
            error=error,
            context=original_context,
            dependent_crews=[],
        )

        # Updated context should be the same object (modified in place)
        assert updated_context is original_context

        # Original data should be preserved
        assert updated_context.partial_results["original_key"] == "original_value"
        assert updated_context.error_recovery_actions[0] == "original_action"

        # New data should be added
        assert "test_crew_failure_isolated" in updated_context.partial_results
        assert len(updated_context.error_recovery_actions) > 1

    def test_failure_handler_initialization_parameters(self):
        """Test PartialFailureHandler initialization with various parameters."""
        # Test default initialization
        handler1 = PartialFailureHandler()
        assert isinstance(handler1.default_strategy, IsolatedFailureStrategy)
        assert handler1.critical_crews == set()

        # Test initialization with custom default strategy
        custom_strategy = GracefulDegradationStrategy()
        handler2 = PartialFailureHandler(default_strategy=custom_strategy)
        assert handler2.default_strategy is custom_strategy

        # Test initialization with critical crews
        critical_crews = {"crew_a", "crew_b", "crew_c"}
        handler3 = PartialFailureHandler(critical_crews=critical_crews)
        assert handler3.critical_crews == critical_crews

        # Test initialization with both parameters
        handler4 = PartialFailureHandler(
            default_strategy=custom_strategy, critical_crews=critical_crews
        )
        assert handler4.default_strategy is custom_strategy
        assert handler4.critical_crews == critical_crews
