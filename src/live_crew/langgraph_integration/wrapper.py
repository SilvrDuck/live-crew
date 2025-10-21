"""LangGraph wrapper that adapts LangGraph workflows to live-crew's EventHandler protocol."""

from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timezone
from collections.abc import MutableMapping

from live_crew.core.models import Event, Action


class LangGraphStateBridge(MutableMapping):
    """Bidirectional state bridge between live-crew context and LangGraph state.

    This class acts as a smart proxy that:
    1. Provides LangGraph workflows with live-crew context data
    2. Tracks state modifications made by LangGraph nodes
    3. Applies state updates back to live-crew's shared context
    4. Handles namespace isolation between different graphs
    5. Manages LangGraph's state schema and reducers
    """

    def __init__(self, live_crew_context: Dict[str, Any], graph_id: str):
        """Initialize the state bridge.

        Args:
            live_crew_context: The live-crew shared context (read-only view)
            graph_id: ID of the graph using this bridge (for namespace isolation)
        """
        self._live_crew_context = live_crew_context.copy()  # Snapshot for reading
        self._graph_id = graph_id
        self._updates: Dict[str, Any] = {}  # Track changes made by LangGraph
        self._deletions: set[str] = set()  # Track deletions made by LangGraph

    def __getitem__(self, key: str) -> Any:
        """Get context value, prioritizing graph updates over original context."""
        if key in self._updates:
            return self._updates[key]
        if key in self._deletions:
            raise KeyError(key)
        return self._live_crew_context[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set context value, tracking as graph update."""
        self._updates[key] = value
        self._deletions.discard(key)  # Remove from deletions if re-added

    def __delitem__(self, key: str) -> None:
        """Delete context value, tracking as graph deletion."""
        if key in self._updates:
            del self._updates[key]
        elif key in self._live_crew_context:
            self._deletions.add(key)
        else:
            raise KeyError(key)

    def __iter__(self):
        """Iterate over all available context keys."""
        all_keys = set(self._live_crew_context.keys()) | set(self._updates.keys())
        return iter(all_keys - self._deletions)

    def __len__(self) -> int:
        """Return the number of context items."""
        return len(
            set(self._live_crew_context.keys())
            | set(self._updates.keys()) - self._deletions
        )

    def get_graph_updates(self) -> Dict[str, Any]:
        """Get all context updates made by the LangGraph workflow.

        Returns:
            Dictionary of context changes (additions/modifications only)
        """
        return self._updates.copy()

    def get_graph_deletions(self) -> set[str]:
        """Get all context deletions made by the LangGraph workflow.

        Returns:
            Set of context keys that were deleted
        """
        return self._deletions.copy()

    def apply_updates_to_live_crew_context(
        self, target_context: Dict[str, Any]
    ) -> None:
        """Apply graph state changes back to live-crew's shared context.

        This method provides namespace isolation by prefixing graph-specific
        updates while allowing global context access.

        Args:
            target_context: The live-crew context to update (mutable)
        """
        # Apply graph updates with namespace isolation
        for key, value in self._updates.items():
            if key.startswith("global_") or key.startswith("shared_"):
                # Allow graphs to update global/shared context directly
                target_context[key] = value
            else:
                # Namespace graph-specific updates to avoid conflicts
                namespaced_key = f"{self._graph_id}_{key}"
                target_context[namespaced_key] = value

        # Apply deletions (with namespace isolation)
        for key in self._deletions:
            if key.startswith("global_") or key.startswith("shared_"):
                target_context.pop(key, None)
            else:
                namespaced_key = f"{self._graph_id}_{key}"
                target_context.pop(namespaced_key, None)

    def to_langgraph_state_format(self, event: Event[Any]) -> Dict[str, Any]:
        """Convert context and event to LangGraph-compatible state format.

        This method formats the context and event data in a way that LangGraph
        workflows can use as initial state. Reserved keys are placed in nested
        structure to avoid conflicts.

        Args:
            event: The triggering event

        Returns:
            Dictionary formatted for LangGraph initial state
        """
        # Reserved keys that conflict with common state fields
        reserved_keys = {"messages", "next", "input", "output"}

        state = {}

        # Flat access for non-conflicting keys
        for key, value in self.items():
            if key not in reserved_keys:
                state[key] = value

        # Event data in nested structure
        state["live_crew_event"] = {
            "kind": event.kind,
            "timestamp": event.ts.isoformat(),
            "stream_id": event.stream_id,
            "payload": event.payload,
        }

        # Context data in nested structure
        state["live_crew_context"] = {
            "all_context": dict(self.items()),
            "graph_id": self._graph_id,
            "updates": self._updates.copy(),
        }

        return state


class LangGraphWrapper:
    """Adapter that enables LangGraph workflows to work with live-crew's event-driven architecture.

    This wrapper converts live-crew Events into LangGraph state inputs, executes the LangGraph workflow,
    and converts the workflow's output state back into live-crew Actions. It serves as the bridge
    between live-crew's orchestration layer and LangGraph's graph-based execution model.

    The wrapper supports both Python-defined graphs and YAML-configured workflows, with built-in
    support for LangGraph's checkpointing and human-in-the-loop features.
    """

    def __init__(
        self,
        graph_id: str,
        langgraph_app: Any,
        triggers: List[str],
        timeout_ms: int = 5000,
        checkpointing: bool = False,
        thread_id_strategy: str = "stream_id",
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
    ):
        """Initialize the LangGraph wrapper.

        Args:
            graph_id: Unique identifier for this graph within live-crew orchestration
            langgraph_app: The compiled LangGraph application to wrap
            triggers: List of event kinds that should trigger this graph
            timeout_ms: Maximum execution time for graph processing
            checkpointing: Enable LangGraph checkpointing for stateful execution
            thread_id_strategy: Strategy for generating thread IDs (stream_id, event_kind, custom)
            interrupt_before: List of node names to interrupt before
            interrupt_after: List of node names to interrupt after
        """
        self.graph_id = graph_id
        self.langgraph_app = langgraph_app
        self.triggers = triggers
        self.timeout_ms = timeout_ms
        self.checkpointing = checkpointing
        self.thread_id_strategy = thread_id_strategy
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []

    @property
    def crew_id(self) -> str:
        """Get the crew identifier (alias for graph_id for EventHandler protocol).

        Returns:
            The graph identifier
        """
        return self.graph_id

    async def handle_event(
        self, event: Event[Any], context: Dict[str, Any]
    ) -> List[Action[Any]]:
        """Handle an event by executing the LangGraph workflow and converting output to actions.

        This method implements the live-crew EventHandler protocol with bidirectional
        state bridge support:
        1. Converting the live-crew Event to LangGraph initial state
        2. Injecting live-crew context as graph state
        3. Executing the LangGraph workflow with state access
        4. Extracting state updates from LangGraph execution
        5. Converting LangGraph output state to live-crew Actions

        Args:
            event: The live-crew event that triggered this graph
            context: Shared context from live-crew's orchestration layer (mutable)

        Returns:
            List of Actions generated from the LangGraph workflow's output

        Raises:
            LangGraphExecutionError: If the LangGraph workflow execution fails
            TimeoutError: If graph execution exceeds timeout_ms
        """
        try:
            # Step 1: Create state bridge for bidirectional data flow
            state_bridge = LangGraphStateBridge(context, self.graph_id)

            # Step 2: Prepare LangGraph initial state with context bridge integration
            initial_state = self._prepare_initial_state(event, state_bridge)

            # Step 3: Prepare execution config (thread_id for checkpointing)
            config = self._prepare_execution_config(event)

            # Step 4: Execute the LangGraph workflow with state access
            # Use invoke for synchronous execution (LangGraph handles async internally)
            final_state = self.langgraph_app.invoke(initial_state, config=config)

            # Step 5: Extract state updates from graph execution
            state_bridge.apply_updates_to_live_crew_context(context)

            # Step 6: Convert LangGraph output state to live-crew Actions
            actions = self._convert_to_actions(final_state, event)

            return actions

        except Exception as e:
            # Convert any LangGraph-specific exceptions to live-crew compatible format
            raise LangGraphExecutionError(
                f"LangGraph workflow '{self.graph_id}' failed during event processing: {str(e)}"
            ) from e

    def _prepare_initial_state(
        self, event: Event[Any], state_bridge: LangGraphStateBridge
    ) -> Dict[str, Any]:
        """Convert live-crew event and context into LangGraph initial state format.

        This method creates the initial state dictionary that will be passed to LangGraph's
        invoke() method. It combines event data and shared context in a format
        that LangGraph workflows can use in their nodes.

        Args:
            event: The triggering event containing payload data
            state_bridge: LangGraph state bridge providing access to live-crew context

        Returns:
            Dictionary of initial state for LangGraph execution
        """
        # Get base state from bridge (includes context and event data)
        initial_state = state_bridge.to_langgraph_state_format(event)

        # Add event payload directly if it's a dictionary (common case)
        if isinstance(event.payload, dict):
            # Merge payload fields directly for easier access in graph nodes
            for key, value in event.payload.items():
                if key not in initial_state:
                    initial_state[key] = value

        # Ensure commonly expected LangGraph fields exist
        if "messages" not in initial_state:
            initial_state["messages"] = []

        return initial_state

    def _prepare_execution_config(self, event: Event[Any]) -> Dict[str, Any]:
        """Prepare LangGraph execution configuration.

        This method creates the config dictionary for LangGraph execution,
        including thread_id for checkpointing and interrupt points.

        Args:
            event: The triggering event

        Returns:
            Dictionary of execution configuration
        """
        config: Dict[str, Any] = {}

        # Configure checkpointing with thread_id
        if self.checkpointing:
            thread_id = self._generate_thread_id(event)
            config["configurable"] = {"thread_id": thread_id}

        # Configure interrupt points for human-in-the-loop
        if self.interrupt_before:
            config["interrupt_before"] = self.interrupt_before
        if self.interrupt_after:
            config["interrupt_after"] = self.interrupt_after

        return config

    def _generate_thread_id(self, event: Event[Any]) -> str:
        """Generate thread ID for LangGraph checkpointing.

        Args:
            event: The triggering event

        Returns:
            Thread ID string
        """
        if self.thread_id_strategy == "stream_id":
            return f"{self.graph_id}_{event.stream_id}"
        elif self.thread_id_strategy == "event_kind":
            return f"{self.graph_id}_{event.kind}"
        else:
            # Default: use stream_id
            return f"{self.graph_id}_{event.stream_id}"

    def _convert_to_actions(
        self, final_state: Dict[str, Any], original_event: Event[Any]
    ) -> List[Action[Any]]:
        """Convert LangGraph final state to live-crew Actions.

        This method analyzes the LangGraph workflow's final state and creates appropriate
        live-crew Actions that can be processed by the orchestration layer.

        Args:
            final_state: The final state from LangGraph execution
            original_event: The event that triggered this graph (for context)

        Returns:
            List of Actions created from the graph output
        """
        actions = []
        current_time = datetime.now(timezone.utc)

        # Extract actions from common LangGraph output patterns
        
        # Pattern 1: Explicit actions list in state
        if "actions" in final_state and isinstance(final_state["actions"], list):
            for action_data in final_state["actions"]:
                if isinstance(action_data, dict):
                    action = Action(
                        ts=current_time,
                        kind=action_data.get("kind", f"{self.graph_id}_action"),
                        stream_id=original_event.stream_id,
                        payload=action_data.get("payload", action_data),
                        ttl_ms=self.timeout_ms,
                    )
                    actions.append(action)

        # Pattern 2: Messages list (common in LangGraph)
        elif "messages" in final_state and isinstance(final_state["messages"], list):
            if final_state["messages"]:
                # Create action from last message
                last_message = final_state["messages"][-1]
                message_content = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
                action = Action(
                    ts=current_time,
                    kind=f"{self.graph_id}_message",
                    stream_id=original_event.stream_id,
                    payload={"message": message_content, "source_event": original_event.kind},
                    ttl_ms=self.timeout_ms,
                )
                actions.append(action)

        # Pattern 3: Output field in state
        elif "output" in final_state:
            action = Action(
                ts=current_time,
                kind=f"{self.graph_id}_output",
                stream_id=original_event.stream_id,
                payload={"output": final_state["output"], "source_event": original_event.kind},
                ttl_ms=self.timeout_ms,
            )
            actions.append(action)

        # Pattern 4: Full state as structured output
        else:
            # Filter out live-crew internal fields
            filtered_state = {
                k: v
                for k, v in final_state.items()
                if not k.startswith("live_crew_")
            }
            
            action = Action(
                ts=current_time,
                kind=f"{self.graph_id}_state",
                stream_id=original_event.stream_id,
                payload={"state": filtered_state, "source_event": original_event.kind},
                ttl_ms=self.timeout_ms,
            )
            actions.append(action)

        return actions


class LangGraphExecutionError(Exception):
    """Exception raised when LangGraph workflow execution fails within live-crew orchestration."""

    pass
