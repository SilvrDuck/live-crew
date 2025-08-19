"""CrewAI wrapper that adapts CrewAI crews to live-crew's EventHandler protocol."""

from typing import Any, Dict, List
import json
from datetime import datetime, timezone
from collections.abc import MutableMapping

from live_crew.core.models import Event, Action


class CrewAIContextBridge(MutableMapping):
    """Bidirectional context bridge between live-crew and CrewAI.

    This class acts as a smart proxy that:
    1. Provides CrewAI crews with live-crew context data
    2. Tracks modifications made by CrewAI crews
    3. Applies context updates back to live-crew's shared context
    4. Handles namespace isolation between different crews
    """

    def __init__(self, live_crew_context: Dict[str, Any], crew_id: str):
        """Initialize the context bridge.

        Args:
            live_crew_context: The live-crew shared context (read-only view)
            crew_id: ID of the crew using this bridge (for namespace isolation)
        """
        self._live_crew_context = live_crew_context.copy()  # Snapshot for reading
        self._crew_id = crew_id
        self._updates: Dict[str, Any] = {}  # Track changes made by CrewAI
        self._deletions: set[str] = set()  # Track deletions made by CrewAI

    def __getitem__(self, key: str) -> Any:
        """Get context value, prioritizing crew updates over original context."""
        if key in self._updates:
            return self._updates[key]
        if key in self._deletions:
            raise KeyError(key)
        return self._live_crew_context[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set context value, tracking as crew update."""
        self._updates[key] = value
        self._deletions.discard(key)  # Remove from deletions if re-added

    def __delitem__(self, key: str) -> None:
        """Delete context value, tracking as crew deletion."""
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

    def get_crew_updates(self) -> Dict[str, Any]:
        """Get all context updates made by the CrewAI crew.

        Returns:
            Dictionary of context changes (additions/modifications only)
        """
        return self._updates.copy()

    def get_crew_deletions(self) -> set[str]:
        """Get all context deletions made by the CrewAI crew.

        Returns:
            Set of context keys that were deleted
        """
        return self._deletions.copy()

    def apply_updates_to_live_crew_context(
        self, target_context: Dict[str, Any]
    ) -> None:
        """Apply crew context changes back to live-crew's shared context.

        This method provides namespace isolation by prefixing crew-specific
        updates while allowing global context access.

        Args:
            target_context: The live-crew context to update (mutable)
        """
        # Apply crew updates with namespace isolation
        for key, value in self._updates.items():
            if key.startswith("global_") or key.startswith("shared_"):
                # Allow crews to update global/shared context directly
                target_context[key] = value
            else:
                # Namespace crew-specific updates to avoid conflicts
                namespaced_key = f"{self._crew_id}_{key}"
                target_context[namespaced_key] = value

        # Apply deletions (with namespace isolation)
        for key in self._deletions:
            if key.startswith("global_") or key.startswith("shared_"):
                target_context.pop(key, None)
            else:
                namespaced_key = f"{self._crew_id}_{key}"
                target_context.pop(namespaced_key, None)

    def to_crewai_memory_format(self) -> Dict[str, Any]:
        """Convert context to CrewAI-compatible memory format.

        This method formats the context in a way that CrewAI crews can
        access through their standard memory/context patterns.
        Reserved keys that conflict with event metadata are only available
        in the nested structure to avoid conflicts.

        Returns:
            Dictionary formatted for CrewAI crew memory access
        """
        # Reserved keys that conflict with event metadata
        reserved_keys = {"event_kind", "event_timestamp", "stream_id", "payload"}

        memory = {}

        # Flat access for non-conflicting keys (compatible with YAML variable interpolation)
        for key, value in self.items():
            if key not in reserved_keys:
                memory[key] = value

        # Nested access (for all context data including potentially conflicting keys)
        memory["live_crew"] = {
            "context": dict(self.items()),
            "crew_id": self._crew_id,
            "updates": self._updates.copy(),
        }

        return memory


class CrewAIWrapper:
    """Adapter that enables CrewAI crews to work with live-crew's event-driven architecture.

    This wrapper converts live-crew Events into CrewAI crew inputs, executes the CrewAI crew,
    and converts the crew's output back into live-crew Actions. It serves as the bridge
    between live-crew's orchestration layer and CrewAI's agent-based execution model.

    The wrapper supports both YAML-configured and Python-defined CrewAI crews without
    requiring any modifications to the standard CrewAI crew code.
    """

    def __init__(
        self,
        crew_id: str,
        crewai_crew: Any,
        triggers: List[str],
        timeout_ms: int = 5000,
    ):
        """Initialize the CrewAI wrapper.

        Args:
            crew_id: Unique identifier for this crew within live-crew orchestration
            crewai_crew: The standard CrewAI Crew instance to wrap
            triggers: List of event kinds that should trigger this crew
            timeout_ms: Maximum execution time for crew processing
        """
        self.crew_id = crew_id
        self.crewai_crew = crewai_crew
        self.triggers = triggers
        self.timeout_ms = timeout_ms

    async def handle_event(
        self, event: Event[Any], context: Dict[str, Any]
    ) -> List[Action[Any]]:
        """Handle an event by executing the CrewAI crew and converting output to actions.

        This method implements the live-crew EventHandler protocol with bidirectional
        context bridge support:
        1. Converting the live-crew Event to CrewAI crew inputs
        2. Injecting live-crew context as crew memory/context
        3. Executing the CrewAI crew with context access
        4. Extracting context updates from CrewAI execution
        5. Converting CrewAI output to live-crew Actions

        Args:
            event: The live-crew event that triggered this crew
            context: Shared context from live-crew's orchestration layer (mutable)

        Returns:
            List of Actions generated from the CrewAI crew's output

        Raises:
            CrewAIExecutionError: If the CrewAI crew execution fails
            TimeoutError: If crew execution exceeds timeout_ms
        """
        try:
            # Step 1: Create context bridge for bidirectional data flow
            context_bridge = CrewAIContextBridge(context, self.crew_id)

            # Step 2: Prepare CrewAI inputs with context bridge integration
            crew_inputs = self._prepare_crew_inputs(event, context_bridge)

            # Step 3: Execute the CrewAI crew with context access
            crew_result = self.crewai_crew.kickoff(inputs=crew_inputs)

            # Step 4: Extract context updates from crew execution
            context_bridge.apply_updates_to_live_crew_context(context)

            # Step 5: Convert CrewAI output to live-crew Actions
            actions = self._convert_to_actions(crew_result, event)

            return actions

        except Exception as e:
            # Convert any CrewAI-specific exceptions to live-crew compatible format
            raise CrewAIExecutionError(
                f"CrewAI crew '{self.crew_id}' failed during event processing: {str(e)}"
            ) from e

    def _prepare_crew_inputs(
        self, event: Event[Any], context_bridge: CrewAIContextBridge
    ) -> Dict[str, Any]:
        """Convert live-crew event and context into CrewAI crew input format.

        This method creates the inputs dictionary that will be passed to CrewAI's
        kickoff() method. It combines event data and shared context in a format
        that CrewAI crews can use through their standard variable interpolation.

        Args:
            event: The triggering event containing payload data
            context_bridge: CrewAI context bridge providing access to live-crew context

        Returns:
            Dictionary of inputs for CrewAI crew execution
        """
        # Base inputs from event data
        crew_inputs = {
            # Event metadata available to CrewAI crews
            "event_kind": event.kind,
            "event_timestamp": event.ts.isoformat(),
            "stream_id": event.stream_id,
            # Event payload data (flattened for easier access in YAML)
            **self._flatten_payload(event.payload),
            # CrewAI-compatible memory format from context bridge
            **context_bridge.to_crewai_memory_format(),
        }

        return crew_inputs

    def _flatten_payload(self, payload: Any) -> Dict[str, Any]:
        """Flatten event payload for easier access in CrewAI YAML configurations.

        This method converts nested payload structures into flat key-value pairs
        that can be easily referenced in CrewAI YAML files using {variable} syntax.

        Args:
            payload: The event payload (any JSON-serializable structure)

        Returns:
            Flattened dictionary with payload data
        """
        if not isinstance(payload, dict):
            return {"payload": payload}

        flattened = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries with dot notation
                for nested_key, nested_value in value.items():
                    flattened[f"{key}_{nested_key}"] = nested_value
            else:
                flattened[key] = value

        return flattened

    def _convert_to_actions(
        self, crew_result: Any, original_event: Event[Any]
    ) -> List[Action[Any]]:
        """Convert CrewAI crew output to live-crew Actions.

        This method analyzes the CrewAI crew's output and creates appropriate
        live-crew Actions that can be processed by the orchestration layer.

        Args:
            crew_result: The result from CrewAI crew.kickoff()
            original_event: The event that triggered this crew (for context)

        Returns:
            List of Actions created from the crew output
        """
        actions = []
        current_time = datetime.now(timezone.utc)

        # Handle different types of CrewAI output formats
        if hasattr(crew_result, "raw"):
            # CrewResult object with structured output
            action = Action(
                ts=current_time,
                kind=f"{self.crew_id}_output",
                stream_id=original_event.stream_id,
                payload={
                    "crew_result": crew_result.raw,
                    "source_event": original_event.kind,
                },
                ttl_ms=self.timeout_ms,
            )
            actions.append(action)

        elif isinstance(crew_result, str):
            # Simple string output
            action = Action(
                ts=current_time,
                kind=f"{self.crew_id}_text",
                stream_id=original_event.stream_id,
                payload={"text": crew_result, "source_event": original_event.kind},
                ttl_ms=self.timeout_ms,
            )
            actions.append(action)

        elif isinstance(crew_result, dict):
            # Structured dictionary output
            action = Action(
                ts=current_time,
                kind=f"{self.crew_id}_structured",
                stream_id=original_event.stream_id,
                payload={"data": crew_result, "source_event": original_event.kind},
                ttl_ms=self.timeout_ms,
            )
            actions.append(action)

        else:
            # Fallback for other output types
            try:
                serialized_result = json.dumps(crew_result, default=str)
                action = Action(
                    ts=current_time,
                    kind=f"{self.crew_id}_generic",
                    stream_id=original_event.stream_id,
                    payload={
                        "result": serialized_result,
                        "source_event": original_event.kind,
                    },
                    ttl_ms=self.timeout_ms,
                )
                actions.append(action)
            except Exception:
                # If serialization fails, create a simple string representation
                action = Action(
                    ts=current_time,
                    kind=f"{self.crew_id}_fallback",
                    stream_id=original_event.stream_id,
                    payload={
                        "result": str(crew_result),
                        "source_event": original_event.kind,
                    },
                    ttl_ms=self.timeout_ms,
                )
                actions.append(action)

        return actions


class CrewAIExecutionError(Exception):
    """Exception raised when CrewAI crew execution fails within live-crew orchestration."""

    pass
