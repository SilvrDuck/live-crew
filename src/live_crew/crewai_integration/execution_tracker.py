"""Execution state tracking for multi-crew orchestration.

This module provides comprehensive state management for crew execution phases,
error tracking, and recovery coordination in multi-crew scenarios.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from live_crew.interfaces.dependency_protocol import ExecutionPhase


class CrewExecutionState(Enum):
    """Represents the current execution state of a crew in the orchestration."""

    PENDING = "pending"  # Crew is waiting for dependencies
    READY = "ready"  # Dependencies satisfied, ready to run
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Execution finished successfully
    FAILED = "failed"  # Execution failed but might retry
    DISABLED = "disabled"  # Disabled due to repeated failures
    CANCELLED = "cancelled"  # Cancelled due to dependency failures


@dataclass
class CrewExecutionLog:
    """Detailed execution log for a single crew."""

    crew_id: str
    state: CrewExecutionState
    phase_index: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output_actions: List[Any] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate execution duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return int(delta.total_seconds() * 1000)
        return None

    def mark_started(self) -> None:
        """Mark crew execution as started."""
        self.start_time = datetime.now(timezone.utc)
        self.state = CrewExecutionState.RUNNING

    def mark_completed(
        self, actions: List[Any], context_updates: Dict[str, Any]
    ) -> None:
        """Mark crew execution as completed successfully."""
        self.end_time = datetime.now(timezone.utc)
        self.state = CrewExecutionState.COMPLETED
        self.output_actions = actions
        self.context_updates = context_updates

    def mark_failed(self, error_message: str) -> None:
        """Mark crew execution as failed."""
        self.end_time = datetime.now(timezone.utc)
        self.state = CrewExecutionState.FAILED
        self.error_message = error_message
        self.retry_count += 1


@dataclass
class ExecutionContext:
    """Comprehensive execution context for multi-crew orchestration."""

    slice_idx: int
    execution_logs: Dict[str, CrewExecutionLog] = field(default_factory=dict)
    phase_execution_order: List[ExecutionPhase] = field(default_factory=list)
    failed_crews: Set[str] = field(default_factory=set)
    disabled_crews: Set[str] = field(default_factory=set)
    completed_crews: Set[str] = field(default_factory=set)
    partial_results: Dict[str, Any] = field(default_factory=dict)
    error_recovery_actions: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    @property
    def total_duration_ms(self) -> Optional[int]:
        """Calculate total orchestration duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return int(delta.total_seconds() * 1000)
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage of completed crews."""
        total_crews = len(self.execution_logs)
        if total_crews == 0:
            return 100.0
        return (len(self.completed_crews) / total_crews) * 100.0

    def get_crew_log(self, crew_id: str) -> CrewExecutionLog:
        """Get or create execution log for a crew."""
        if crew_id not in self.execution_logs:
            # Find the phase this crew belongs to
            phase_index = 0
            for phase in self.phase_execution_order:
                if crew_id in phase.crews:
                    phase_index = phase.phase_index
                    break

            self.execution_logs[crew_id] = CrewExecutionLog(
                crew_id=crew_id,
                state=CrewExecutionState.PENDING,
                phase_index=phase_index,
            )
        return self.execution_logs[crew_id]


class ExecutionTracker:
    """Production-ready execution tracker for multi-crew orchestration state management.

    This tracker provides comprehensive monitoring of crew execution states,
    error tracking, recovery coordination, and performance metrics collection.

    Example:
        tracker = ExecutionTracker()
        context = tracker.create_execution_context(slice_idx=0, phases=execution_phases)

        # During execution
        tracker.mark_crew_started(context, "crew_analytics")
        tracker.mark_crew_completed(context, "crew_analytics", actions, context_updates)

        # Get metrics
        metrics = tracker.get_execution_metrics(context)
    """

    def __init__(self, failure_threshold: int = 3):
        """Initialize execution tracker with failure management settings.

        Args:
            failure_threshold: Number of failures before crew is disabled
        """
        self._failure_threshold = failure_threshold
        self._global_failure_counts: Dict[str, int] = {}

    def create_execution_context(
        self, slice_idx: int, phases: List[ExecutionPhase]
    ) -> ExecutionContext:
        """Create a new execution context for orchestration tracking.

        Args:
            slice_idx: Current time slice index
            phases: Execution phases from dependency resolution

        Returns:
            ExecutionContext ready for crew execution tracking
        """
        context = ExecutionContext(slice_idx=slice_idx, phase_execution_order=phases)

        # Initialize crew logs for all crews in all phases
        for phase in phases:
            for crew_id in phase.crews:
                log = context.get_crew_log(crew_id)
                log.state = (
                    CrewExecutionState.READY
                    if phase.phase_index == 0
                    else CrewExecutionState.PENDING
                )

        return context

    def mark_crew_started(self, context: ExecutionContext, crew_id: str) -> None:
        """Mark a crew as started execution.

        Args:
            context: Current execution context
            crew_id: ID of crew starting execution
        """
        log = context.get_crew_log(crew_id)
        log.mark_started()

    def mark_crew_completed(
        self,
        context: ExecutionContext,
        crew_id: str,
        actions: List[Any],
        context_updates: Dict[str, Any],
    ) -> None:
        """Mark a crew as completed successfully.

        Args:
            context: Current execution context
            crew_id: ID of completed crew
            actions: Actions produced by crew
            context_updates: Context updates made by crew
        """
        log = context.get_crew_log(crew_id)
        log.mark_completed(actions, context_updates)
        context.completed_crews.add(crew_id)

        # Reset failure count on successful completion
        if crew_id in self._global_failure_counts:
            del self._global_failure_counts[crew_id]

        # Update ready state for crews in next phases
        self._update_dependent_crew_states(context, crew_id)

    def mark_crew_failed(
        self,
        context: ExecutionContext,
        crew_id: str,
        error_message: str,
        should_disable: bool = False,
    ) -> bool:
        """Mark a crew as failed with error handling.

        Args:
            context: Current execution context
            crew_id: ID of failed crew
            error_message: Error description
            should_disable: Force disable crew regardless of failure count

        Returns:
            True if crew was disabled due to failures, False if still eligible for retry
        """
        log = context.get_crew_log(crew_id)
        log.mark_failed(error_message)
        context.failed_crews.add(crew_id)

        # Track global failure count
        self._global_failure_counts[crew_id] = (
            self._global_failure_counts.get(crew_id, 0) + 1
        )

        # Check if crew should be disabled
        if (
            should_disable
            or self._global_failure_counts[crew_id] >= self._failure_threshold
        ):
            self._disable_crew(context, crew_id, error_message)
            return True

        return False

    def mark_crew_cancelled(
        self, context: ExecutionContext, crew_id: str, reason: str
    ) -> None:
        """Mark a crew as cancelled due to dependency failures.

        Args:
            context: Current execution context
            crew_id: ID of cancelled crew
            reason: Cancellation reason
        """
        log = context.get_crew_log(crew_id)
        log.state = CrewExecutionState.CANCELLED
        log.error_message = f"Cancelled: {reason}"
        context.error_recovery_actions.append(f"Crew '{crew_id}' cancelled: {reason}")

    def get_ready_crews(self, context: ExecutionContext, phase_index: int) -> List[str]:
        """Get crews ready for execution in specified phase.

        Args:
            context: Current execution context
            phase_index: Phase to get ready crews from

        Returns:
            List of crew IDs ready for execution
        """
        ready_crews = []

        if phase_index < len(context.phase_execution_order):
            phase = context.phase_execution_order[phase_index]
            for crew_id in phase.crews:
                log = context.get_crew_log(crew_id)
                if log.state == CrewExecutionState.READY:
                    ready_crews.append(crew_id)

        return ready_crews

    def get_execution_metrics(self, context: ExecutionContext) -> Dict[str, Any]:
        """Get comprehensive execution metrics for monitoring.

        Args:
            context: Execution context to analyze

        Returns:
            Dictionary of execution metrics and statistics
        """
        total_crews = len(context.execution_logs)
        running_crews = len(
            [
                log
                for log in context.execution_logs.values()
                if log.state == CrewExecutionState.RUNNING
            ]
        )

        # Calculate average execution time for completed crews
        completed_durations = [
            log.duration_ms
            for log in context.execution_logs.values()
            if log.state == CrewExecutionState.COMPLETED and log.duration_ms
        ]
        avg_duration_ms = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations
            else 0
        )

        # Count crews by state
        state_counts = {}
        for state in CrewExecutionState:
            state_counts[state.value] = len(
                [log for log in context.execution_logs.values() if log.state == state]
            )

        return {
            "slice_idx": context.slice_idx,
            "total_crews": total_crews,
            "completed_crews": len(context.completed_crews),
            "failed_crews": len(context.failed_crews),
            "disabled_crews": len(context.disabled_crews),
            "running_crews": running_crews,
            "success_rate_percent": context.success_rate,
            "total_duration_ms": context.total_duration_ms,
            "average_crew_duration_ms": avg_duration_ms,
            "phase_count": len(context.phase_execution_order),
            "error_recovery_actions": len(context.error_recovery_actions),
            "state_distribution": state_counts,
        }

    def finalize_execution(self, context: ExecutionContext) -> None:
        """Finalize execution context and record end time.

        Args:
            context: Execution context to finalize
        """
        context.end_time = datetime.now(timezone.utc)

    def _disable_crew(
        self, context: ExecutionContext, crew_id: str, reason: str
    ) -> None:
        """Disable a crew due to repeated failures.

        Args:
            context: Current execution context
            crew_id: ID of crew to disable
            reason: Reason for disabling
        """
        log = context.get_crew_log(crew_id)
        log.state = CrewExecutionState.DISABLED
        context.disabled_crews.add(crew_id)
        context.error_recovery_actions.append(
            f"Crew '{crew_id}' disabled after {self._failure_threshold} failures: {reason}"
        )

    def _update_dependent_crew_states(
        self, context: ExecutionContext, completed_crew_id: str
    ) -> None:
        """Update state of crews that were waiting for the completed crew.

        Args:
            context: Current execution context
            completed_crew_id: ID of crew that just completed
        """
        # This is a simplified implementation. In a full dependency-aware system,
        # this would check actual dependency relationships from the resolver
        # and only mark crews as ready when ALL their dependencies are satisfied.

        # For now, we mark crews in the next phase as ready when current phase crews complete
        completed_crew_phase = None
        for log in context.execution_logs.values():
            if log.crew_id == completed_crew_id:
                completed_crew_phase = log.phase_index
                break

        if completed_crew_phase is not None:
            next_phase_index = completed_crew_phase + 1
            if next_phase_index < len(context.phase_execution_order):
                # Check if all crews in the current phase are done
                current_phase_crews = context.phase_execution_order[
                    completed_crew_phase
                ].crews
                all_current_phase_done = all(
                    context.execution_logs[crew_id].state
                    in {
                        CrewExecutionState.COMPLETED,
                        CrewExecutionState.FAILED,
                        CrewExecutionState.DISABLED,
                        CrewExecutionState.CANCELLED,
                    }
                    for crew_id in current_phase_crews
                    if crew_id in context.execution_logs
                )

                if all_current_phase_done:
                    # Mark next phase crews as ready
                    next_phase_crews = context.phase_execution_order[
                        next_phase_index
                    ].crews
                    for crew_id in next_phase_crews:
                        log = context.get_crew_log(crew_id)
                        if log.state == CrewExecutionState.PENDING:
                            log.state = CrewExecutionState.READY
