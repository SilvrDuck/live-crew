"""Tests for LangGraph integration."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from live_crew.core.models import Event, Action
from live_crew.langgraph_integration.models import GraphRuntimeConfig
from live_crew.langgraph_integration.wrapper import (
    LangGraphWrapper,
    LangGraphStateBridge,
    LangGraphExecutionError,
)
from live_crew.langgraph_integration.loader import (
    LangGraphLoader,
    LangGraphLoadError,
    LangGraphConfigError,
)


class TestGraphRuntimeConfig:
    """Tests for GraphRuntimeConfig model."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = GraphRuntimeConfig(
            graph="test_graph",
            triggers=["test_event"],
            timeout_ms=5000,
        )
        assert config.graph == "test_graph"
        assert config.triggers == ["test_event"]
        assert config.timeout_ms == 5000
        assert config.checkpointing is False

    def test_config_with_checkpointing(self):
        """Test configuration with checkpointing enabled."""
        config = GraphRuntimeConfig(
            graph="test_graph",
            triggers=["test_event"],
            checkpointing=True,
            thread_id_strategy="event_kind",
        )
        assert config.checkpointing is True
        assert config.thread_id_strategy == "event_kind"

    def test_config_with_interrupts(self):
        """Test configuration with interrupt points."""
        config = GraphRuntimeConfig(
            graph="test_graph",
            triggers=["test_event"],
            interrupt_before=["node1", "node2"],
            interrupt_after=["node3"],
        )
        assert config.interrupt_before == ["node1", "node2"]
        assert config.interrupt_after == ["node3"]

    def test_invalid_trigger_pattern(self):
        """Test that invalid trigger patterns are rejected."""
        with pytest.raises(ValueError):
            GraphRuntimeConfig(
                graph="test_graph", triggers=["invalid-trigger!"]  # Invalid character
            )


class TestLangGraphStateBridge:
    """Tests for LangGraphStateBridge."""

    def test_bridge_initialization(self):
        """Test state bridge initialization."""
        context = {"key1": "value1", "key2": "value2"}
        bridge = LangGraphStateBridge(context, "test_graph")

        assert bridge["key1"] == "value1"
        assert bridge["key2"] == "value2"
        assert len(bridge) == 2

    def test_bridge_updates(self):
        """Test tracking updates to state."""
        context = {"existing": "value"}
        bridge = LangGraphStateBridge(context, "test_graph")

        bridge["new_key"] = "new_value"
        bridge["existing"] = "updated"

        updates = bridge.get_graph_updates()
        assert updates["new_key"] == "new_value"
        assert updates["existing"] == "updated"

    def test_bridge_deletions(self):
        """Test tracking deletions from state."""
        context = {"key1": "value1", "key2": "value2"}
        bridge = LangGraphStateBridge(context, "test_graph")

        del bridge["key1"]

        deletions = bridge.get_graph_deletions()
        assert "key1" in deletions
        assert "key2" not in deletions

    def test_to_langgraph_state_format(self):
        """Test conversion to LangGraph state format."""
        context = {"data": "value"}
        bridge = LangGraphStateBridge(context, "test_graph")

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_1",
            payload={"input": "test"},
        )

        state = bridge.to_langgraph_state_format(event)

        assert "data" in state
        assert "live_crew_event" in state
        assert state["live_crew_event"]["kind"] == "test_event"
        assert "live_crew_context" in state


class TestLangGraphWrapper:
    """Tests for LangGraphWrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        mock_app = Mock()
        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
            timeout_ms=5000,
        )

        assert wrapper.graph_id == "test_graph"
        assert wrapper.crew_id == "test_graph"  # Alias for EventHandler protocol
        assert wrapper.triggers == ["test_event"]
        assert wrapper.timeout_ms == 5000

    @pytest.mark.asyncio
    async def test_handle_event_with_actions(self):
        """Test handling event that produces actions."""
        # Mock LangGraph app that returns state with actions
        mock_app = Mock()
        mock_app.invoke.return_value = {
            "actions": [
                {"kind": "test_action", "payload": {"result": "success"}},
            ]
        }

        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_1",
            payload={"input": "test"},
        )

        context: Dict[str, Any] = {}
        actions = await wrapper.handle_event(event, context)

        assert len(actions) == 1
        assert actions[0].kind == "test_action"
        assert actions[0].stream_id == "stream_1"

    @pytest.mark.asyncio
    async def test_handle_event_with_messages(self):
        """Test handling event that produces messages."""
        # Mock LangGraph app that returns state with messages
        mock_message = Mock()
        mock_message.content = "Test message content"

        mock_app = Mock()
        mock_app.invoke.return_value = {"messages": [mock_message]}

        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_1",
            payload={"input": "test"},
        )

        context: Dict[str, Any] = {}
        actions = await wrapper.handle_event(event, context)

        assert len(actions) == 1
        assert actions[0].kind == "test_graph_message"
        assert "Test message content" in str(actions[0].payload)

    @pytest.mark.asyncio
    async def test_handle_event_with_output(self):
        """Test handling event with output field."""
        mock_app = Mock()
        mock_app.invoke.return_value = {"output": "Final output"}

        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_1",
            payload={"input": "test"},
        )

        context: Dict[str, Any] = {}
        actions = await wrapper.handle_event(event, context)

        assert len(actions) == 1
        assert actions[0].kind == "test_graph_output"

    @pytest.mark.asyncio
    async def test_handle_event_execution_error(self):
        """Test that execution errors are properly wrapped."""
        mock_app = Mock()
        mock_app.invoke.side_effect = Exception("Graph execution failed")

        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_1",
            payload={"input": "test"},
        )

        context: Dict[str, Any] = {}

        with pytest.raises(LangGraphExecutionError):
            await wrapper.handle_event(event, context)

    def test_generate_thread_id_stream_id_strategy(self):
        """Test thread ID generation with stream_id strategy."""
        mock_app = Mock()
        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
            checkpointing=True,
            thread_id_strategy="stream_id",
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream_123",
            payload={},
        )

        thread_id = wrapper._generate_thread_id(event)
        assert thread_id == "test_graph_stream_123"

    def test_generate_thread_id_event_kind_strategy(self):
        """Test thread ID generation with event_kind strategy."""
        mock_app = Mock()
        wrapper = LangGraphWrapper(
            graph_id="test_graph",
            langgraph_app=mock_app,
            triggers=["test_event"],
            checkpointing=True,
            thread_id_strategy="event_kind",
        )

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="data_request",
            stream_id="stream_123",
            payload={},
        )

        thread_id = wrapper._generate_thread_id(event)
        assert thread_id == "test_graph_data_request"


class TestLangGraphLoader:
    """Tests for LangGraphLoader."""

    def test_load_python_graph_success(self):
        """Test loading graph from Python definition."""
        mock_app = Mock()
        mock_app.invoke = Mock()

        runtime_config = {
            "triggers": ["test_event"],
            "timeout_ms": 5000,
            "checkpointing": False,
        }

        wrapper = LangGraphLoader.load_python_graph(
            graph_id="test_graph", langgraph_app=mock_app, runtime_config=runtime_config
        )

        assert wrapper.graph_id == "test_graph"
        assert wrapper.triggers == ["test_event"]
        assert wrapper.timeout_ms == 5000

    def test_load_python_graph_invalid_config(self):
        """Test that invalid config raises error."""
        mock_app = Mock()

        # Missing required triggers field
        runtime_config = {"timeout_ms": 5000}

        with pytest.raises(LangGraphConfigError):
            LangGraphLoader.load_python_graph(
                graph_id="test_graph",
                langgraph_app=mock_app,
                runtime_config=runtime_config,
            )
