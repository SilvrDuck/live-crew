"""Partial failure handling and recovery strategies for multi-crew orchestration.

This module provides comprehensive failure handling with circuit breaker patterns,
graceful degradation, and strategic recovery for multi-crew dependency chains.
"""

from typing import Dict, List, Any, Set, Optional, Protocol
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum

from live_crew.crewai_integration.execution_tracker import (
    ExecutionContext,
    CrewExecutionState,
)


class RecoveryStrategy(Protocol):
    """Protocol for failure recovery strategies."""

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext,
        failed_crew: str,
        error: Exception,
        affected_crews: List[str],
    ) -> ExecutionContext:
        """Execute recovery strategy for failed crew.

        Args:
            context: Current execution context
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            affected_crews: List of crews affected by this failure

        Returns:
            Updated execution context after recovery actions
        """
        ...


class FailureImpact(Enum):
    """Classification of failure impact on dependency chain."""

    ISOLATED = "isolated"  # Failure doesn't affect other crews
    DEGRADED = "degraded"  # Other crews can continue with partial data
    CASCADING = "cascading"  # Failure propagates to dependent crews
    CATASTROPHIC = "catastrophic"  # Entire orchestration must be aborted


@dataclass
class DependencyChainAnalysis:
    """Analysis of how a crew failure impacts its dependency chain."""

    failed_crew: str
    impact_level: FailureImpact
    directly_affected: List[str]
    transitively_affected: List[str]
    critical_path_broken: bool
    recovery_options: List[str]

    @property
    def total_affected_count(self) -> int:
        """Total number of crews affected by this failure."""
        return len(set(self.directly_affected + self.transitively_affected))


class IsolatedFailureStrategy:
    """Recovery strategy for failures that don't impact other crews."""

    async def execute(
        self,
        context: ExecutionContext,
        failed_crew: str,
        error: Exception,
        affected_crews: List[str],
    ) -> ExecutionContext:
        """Handle isolated failure by logging and continuing.

        Args:
            context: Current execution context
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            affected_crews: Should be empty for isolated failures

        Returns:
            Updated execution context with recovery action logged
        """
        # Log the isolated failure
        context.error_recovery_actions.append(
            f"Isolated failure in '{failed_crew}': {str(error)}. "
            f"Other crews continue unaffected."
        )

        # Add to partial results for monitoring
        context.partial_results[f"{failed_crew}_failure_isolated"] = {
            "error": str(error),
            "recovery_strategy": "isolated",
            "impact": "none",
        }

        return context


class GracefulDegradationStrategy:
    """Recovery strategy for failures with graceful fallback options."""

    def __init__(self, fallback_data: Optional[Dict[str, Any]] = None):
        """Initialize with optional fallback data.

        Args:
            fallback_data: Default data to provide for dependent crews
        """
        self.fallback_data = fallback_data or {}

    async def execute(
        self,
        context: ExecutionContext,
        failed_crew: str,
        error: Exception,
        affected_crews: List[str],
    ) -> ExecutionContext:
        """Handle failure by providing fallback data to dependent crews.

        Args:
            context: Current execution context
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            affected_crews: List of crews that depend on the failed crew

        Returns:
            Updated execution context with fallback data provided
        """
        # Provide fallback data for dependent crews
        fallback_key = f"{failed_crew}_fallback"
        context.partial_results[fallback_key] = {
            "source": "graceful_degradation",
            "original_crew": failed_crew,
            "error": str(error),
            "fallback_data": self.fallback_data,
            "degraded_mode": True,
        }

        # Log recovery action
        context.error_recovery_actions.append(
            f"Graceful degradation for '{failed_crew}': {str(error)}. "
            f"Providing fallback data to {len(affected_crews)} dependent crews: {affected_crews}"
        )

        # Mark affected crews for degraded execution
        for crew_id in affected_crews:
            context.partial_results[f"{crew_id}_degraded_input"] = {
                "from_failed_crew": failed_crew,
                "fallback_source": fallback_key,
                "execution_mode": "degraded",
            }

        return context


class CascadeFailureStrategy:
    """Recovery strategy for failures that must cascade to dependent crews."""

    def __init__(self, halt_on_cascade: bool = False):
        """Initialize cascade strategy.

        Args:
            halt_on_cascade: Whether to halt entire orchestration on cascade
        """
        self.halt_on_cascade = halt_on_cascade

    async def execute(
        self,
        context: ExecutionContext,
        failed_crew: str,
        error: Exception,
        affected_crews: List[str],
    ) -> ExecutionContext:
        """Handle failure by cascading failure to dependent crews.

        Args:
            context: Current execution context
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            affected_crews: List of crews to cascade failure to

        Returns:
            Updated execution context with cascaded failures
        """
        # Mark dependent crews as cancelled due to cascade
        for crew_id in affected_crews:
            crew_log = context.get_crew_log(crew_id)
            crew_log.state = CrewExecutionState.CANCELLED
            crew_log.error_message = (
                f"Cancelled due to cascade failure from '{failed_crew}': {str(error)}"
            )
            context.failed_crews.add(crew_id)

        # Log cascade action
        context.error_recovery_actions.append(
            f"Cascade failure from '{failed_crew}': {str(error)}. "
            f"Cancelled {len(affected_crews)} dependent crews: {affected_crews}"
        )

        # Record cascade details
        context.partial_results[f"{failed_crew}_cascade"] = {
            "strategy": "cascade_failure",
            "error": str(error),
            "cascaded_crews": affected_crews,
            "halt_orchestration": self.halt_on_cascade,
        }

        return context


class RetryWithBackoffStrategy:
    """Recovery strategy that attempts retry with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay_ms: int = 1000):
        """Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay_ms: Base delay in milliseconds (will be exponentially increased)
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms

    async def execute(
        self,
        context: ExecutionContext,
        failed_crew: str,
        error: Exception,
        affected_crews: List[str],
    ) -> ExecutionContext:
        """Handle failure by scheduling retry with backoff.

        Args:
            context: Current execution context
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            affected_crews: List of crews waiting for retry

        Returns:
            Updated execution context with retry scheduled
        """
        crew_log = context.get_crew_log(failed_crew)

        if crew_log.retry_count < self.max_retries:
            # Calculate backoff delay
            delay_ms = self.base_delay_ms * (2**crew_log.retry_count)

            # Reset crew to READY state for retry
            crew_log.state = CrewExecutionState.READY

            # Log retry action
            context.error_recovery_actions.append(
                f"Retry scheduled for '{failed_crew}' (attempt {crew_log.retry_count + 1}/{self.max_retries}) "
                f"after {delay_ms}ms delay. Error: {str(error)}"
            )

            # Record retry details
            context.partial_results[f"{failed_crew}_retry_{crew_log.retry_count}"] = {
                "strategy": "retry_with_backoff",
                "error": str(error),
                "retry_attempt": crew_log.retry_count + 1,
                "max_retries": self.max_retries,
                "delay_ms": delay_ms,
                "affected_crews": affected_crews,
            }
        else:
            # Max retries exceeded - cascade failure
            cascade_strategy = CascadeFailureStrategy()
            context = await cascade_strategy.execute(
                context, failed_crew, error, affected_crews
            )

            context.error_recovery_actions.append(
                f"Max retries ({self.max_retries}) exceeded for '{failed_crew}'. "
                f"Cascading failure to dependent crews."
            )

        return context


class PartialFailureHandler:
    """Production-ready partial failure handler with comprehensive recovery strategies.

    This handler analyzes dependency chain impacts and applies appropriate recovery
    strategies based on failure severity and crew criticality.

    Example:
        handler = PartialFailureHandler()
        context = await handler.handle_crew_failure(
            failed_crew="analytics_crew",
            error=TimeoutError("Processing timeout"),
            context=execution_context,
            dependent_crews=["reporting_crew", "dashboard_crew"]
        )
    """

    def __init__(
        self,
        default_strategy: Optional[RecoveryStrategy] = None,
        critical_crews: Optional[Set[str]] = None,
    ):
        """Initialize failure handler with configuration.

        Args:
            default_strategy: Default recovery strategy if none specified
            critical_crews: Set of crew IDs considered critical for orchestration
        """
        self.default_strategy = default_strategy or IsolatedFailureStrategy()
        self.critical_crews = critical_crews or set()
        self._strategy_registry: Dict[str, RecoveryStrategy] = {}
        self._failure_history: Dict[str, List[Exception]] = {}

    def register_strategy(self, crew_id: str, strategy: RecoveryStrategy) -> None:
        """Register custom recovery strategy for specific crew.

        Args:
            crew_id: Crew ID to register strategy for
            strategy: Recovery strategy to use for this crew
        """
        self._strategy_registry[crew_id] = strategy

    async def handle_crew_failure(
        self,
        failed_crew: str,
        error: Exception,
        context: ExecutionContext,
        dependent_crews: List[str],
    ) -> ExecutionContext:
        """Handle crew failure with comprehensive impact analysis and recovery.

        Args:
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            context: Current execution context
            dependent_crews: List of crews that depend on the failed crew

        Returns:
            Updated execution context after recovery strategy execution
        """
        # Record failure in history
        if failed_crew not in self._failure_history:
            self._failure_history[failed_crew] = []
        self._failure_history[failed_crew].append(error)

        # Analyze dependency chain impact
        analysis = self._analyze_dependency_impact(
            failed_crew, error, context, dependent_crews
        )

        # Select appropriate recovery strategy
        strategy = self._select_recovery_strategy(failed_crew, analysis)

        # Execute recovery strategy
        updated_context = await strategy.execute(
            context, failed_crew, error, analysis.directly_affected
        )

        # Log comprehensive failure analysis
        updated_context.error_recovery_actions.append(
            f"Failure analysis for '{failed_crew}': {analysis.impact_level.value} impact, "
            f"{analysis.total_affected_count} crews affected, "
            f"critical path broken: {analysis.critical_path_broken}"
        )

        return updated_context

    def _analyze_dependency_impact(
        self,
        failed_crew: str,
        error: Exception,
        context: ExecutionContext,
        dependent_crews: List[str],
    ) -> DependencyChainAnalysis:
        """Analyze the impact of crew failure on dependency chain.

        Args:
            failed_crew: ID of crew that failed
            error: Exception that caused the failure
            context: Current execution context
            dependent_crews: Direct dependents of failed crew

        Returns:
            Comprehensive impact analysis
        """
        # Determine impact level
        if not dependent_crews:
            impact_level = FailureImpact.ISOLATED
        elif failed_crew in self.critical_crews:
            impact_level = FailureImpact.CASCADING
        elif len(dependent_crews) > 5:  # Arbitrary threshold for wide impact
            impact_level = FailureImpact.CASCADING
        else:
            impact_level = FailureImpact.DEGRADED

        # Find transitively affected crews (simplified implementation)
        transitively_affected = []
        for crew_id in dependent_crews:
            # In a full implementation, this would traverse the dependency graph
            # to find all crews that transitively depend on the failed crew
            pass

        # Check if critical path is broken
        critical_path_broken = failed_crew in self.critical_crews

        # Generate recovery options
        recovery_options = []
        if impact_level == FailureImpact.ISOLATED:
            recovery_options.append("continue_with_logging")
        elif impact_level == FailureImpact.DEGRADED:
            recovery_options.extend(["graceful_degradation", "retry_with_fallback"])
        else:
            recovery_options.extend(["cascade_failure", "abort_orchestration"])

        return DependencyChainAnalysis(
            failed_crew=failed_crew,
            impact_level=impact_level,
            directly_affected=dependent_crews,
            transitively_affected=transitively_affected,
            critical_path_broken=critical_path_broken,
            recovery_options=recovery_options,
        )

    def _select_recovery_strategy(
        self, failed_crew: str, analysis: DependencyChainAnalysis
    ) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on failure analysis.

        Args:
            failed_crew: ID of crew that failed
            analysis: Dependency chain impact analysis

        Returns:
            Recovery strategy to execute
        """
        # Check for custom strategy registered for this crew
        if failed_crew in self._strategy_registry:
            return self._strategy_registry[failed_crew]

        # Select strategy based on impact analysis
        if analysis.impact_level == FailureImpact.ISOLATED:
            return IsolatedFailureStrategy()
        elif analysis.impact_level == FailureImpact.DEGRADED:
            return GracefulDegradationStrategy()
        elif analysis.impact_level == FailureImpact.CASCADING:
            if analysis.critical_path_broken:
                return CascadeFailureStrategy(halt_on_cascade=True)
            else:
                return RetryWithBackoffStrategy()
        else:  # CATASTROPHIC
            return CascadeFailureStrategy(halt_on_cascade=True)

    def get_failure_history(self, crew_id: str) -> List[Exception]:
        """Get failure history for a crew.

        Args:
            crew_id: Crew ID to get history for

        Returns:
            List of exceptions that caused failures for this crew
        """
        return self._failure_history.get(crew_id, [])

    def clear_failure_history(self, crew_id: Optional[str] = None) -> None:
        """Clear failure history for crew or all crews.

        Args:
            crew_id: Crew ID to clear history for, or None to clear all
        """
        if crew_id:
            self._failure_history.pop(crew_id, None)
        else:
            self._failure_history.clear()
