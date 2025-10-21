"""Tests for MQTT transport implementations."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from live_crew.core.models import Action, Event
from live_crew.transports.mqtt import MQTTActionTransport, MQTTEventTransport


class TestMQTTEventTransport:
    """Test suite for MQTT event transport."""

    @pytest.fixture
    def mock_mqtt_client(self):
        """Create mock MQTT client."""
        with patch("live_crew.transports.mqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock context manager
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            yield mock_client

    def test_init_default_values(self):
        """Test initialization with default values."""
        transport = MQTTEventTransport(hostname="test-broker")

        assert transport.hostname == "test-broker"
        assert transport.port == 1883
        assert transport.topic_prefix == "live"
        assert transport.username is None
        assert transport.password is None
        assert transport.qos == 1
        assert transport.client_id is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        transport = MQTTEventTransport(
            hostname="mqtt.example.com",
            port=8883,
            topic_prefix="production/live",
            username="test_user",
            password="test_pass",
            qos=2,
            client_id="test-client-1",
        )

        assert transport.hostname == "mqtt.example.com"
        assert transport.port == 8883
        assert transport.topic_prefix == "production/live"
        assert transport.username == "test_user"
        assert transport.password == "test_pass"
        assert transport.qos == 2
        assert transport.client_id == "test-client-1"

    def test_init_invalid_qos(self):
        """Test initialization with invalid QoS level."""
        with pytest.raises(ValueError, match="Invalid QoS level"):
            MQTTEventTransport(hostname="test-broker", qos=3)

        with pytest.raises(ValueError, match="Invalid QoS level"):
            MQTTEventTransport(hostname="test-broker", qos=-1)

    async def test_publish_event(self, mock_mqtt_client):
        """Test publishing an event to MQTT broker."""
        transport = MQTTEventTransport(
            hostname="test-broker", topic_prefix="test"
        )

        event = Event(
            ts=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            kind="test_event",
            stream_id="test_stream",
            payload={"data": "test"},
        )

        await transport.publish_event(event)

        # Verify publish was called once
        mock_mqtt_client.publish.assert_called_once()

        # Verify call arguments
        call_args = mock_mqtt_client.publish.call_args
        assert call_args.kwargs["topic"] == "test/test_stream/events"
        assert call_args.kwargs["qos"] == 1

        # Verify payload contains event data
        payload = json.loads(call_args.kwargs["payload"])
        assert payload["kind"] == "test_event"
        assert payload["stream_id"] == "test_stream"
        assert payload["payload"] == {"data": "test"}
        assert "ts" in payload

    async def test_publish_event_with_custom_qos(self, mock_mqtt_client):
        """Test publishing event with custom QoS."""
        transport = MQTTEventTransport(hostname="test-broker", qos=2)

        event = Event(
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="stream1",
            payload={"value": 42},
        )

        await transport.publish_event(event)

        call_args = mock_mqtt_client.publish.call_args
        assert call_args.kwargs["qos"] == 2

    async def test_subscribe_events(self, mock_mqtt_client):
        """Test subscribing to events from MQTT broker."""
        transport = MQTTEventTransport(
            hostname="test-broker", topic_prefix="test"
        )

        # Mock messages to be received
        mock_message = MagicMock()
        mock_message.topic.value = "test/stream1/events"
        mock_message.payload = json.dumps(
            {
                "ts": "2025-01-15T10:30:00+00:00",
                "kind": "event_received",
                "stream_id": "stream1",
                "payload": {"data": "test_data"},
            }
        ).encode()

        # Mock async iterator
        async def mock_messages():
            yield mock_message

        mock_mqtt_client.messages = mock_messages()

        # Subscribe and collect events
        events = []
        async for event in transport.subscribe_events():
            events.append(event)
            break  # Only process one event for test

        # Verify subscription
        mock_mqtt_client.subscribe.assert_called_once_with("test/+/events", qos=1)

        # Verify event was parsed correctly
        assert len(events) == 1
        assert events[0].kind == "event_received"
        assert events[0].stream_id == "stream1"
        assert events[0].payload == {"data": "test_data"}

    async def test_subscribe_events_invalid_json(self, mock_mqtt_client):
        """Test handling of invalid JSON in subscription."""
        transport = MQTTEventTransport(hostname="test-broker")

        # Mock invalid message
        mock_message = MagicMock()
        mock_message.topic.value = "live/stream1/events"
        mock_message.payload = b"invalid json"

        # Mock valid message after invalid one
        valid_message = MagicMock()
        valid_message.topic.value = "live/stream1/events"
        valid_message.payload = json.dumps(
            {
                "ts": "2025-01-15T10:30:00+00:00",
                "kind": "valid_event",
                "stream_id": "stream1",
                "payload": {},
            }
        ).encode()

        async def mock_messages():
            yield mock_message
            yield valid_message

        mock_mqtt_client.messages = mock_messages()

        # Subscribe - should skip invalid and process valid
        events = []
        async for event in transport.subscribe_events():
            events.append(event)
            if len(events) >= 1:
                break

        # Should have received the valid event
        assert len(events) == 1
        assert events[0].kind == "valid_event"

    async def test_close(self, mock_mqtt_client):
        """Test closing MQTT connection."""
        transport = MQTTEventTransport(hostname="test-broker")

        # Create client connection
        await transport._get_client()
        assert transport._client is not None

        # Close connection
        await transport.close()

        # Verify __aexit__ was called
        mock_mqtt_client.__aexit__.assert_called_once()
        assert transport._client is None

    async def test_close_without_connection(self):
        """Test closing when no connection exists."""
        transport = MQTTEventTransport(hostname="test-broker")

        # Should not raise error
        await transport.close()
        assert transport._client is None


class TestMQTTActionTransport:
    """Test suite for MQTT action transport."""

    @pytest.fixture
    def mock_mqtt_client(self):
        """Create mock MQTT client."""
        with patch("live_crew.transports.mqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock context manager
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            yield mock_client

    def test_init_default_values(self):
        """Test initialization with default values."""
        transport = MQTTActionTransport(hostname="test-broker")

        assert transport.hostname == "test-broker"
        assert transport.port == 1883
        assert transport.topic_prefix == "live"
        assert transport.qos == 1

    def test_init_invalid_qos(self):
        """Test initialization with invalid QoS level."""
        with pytest.raises(ValueError, match="Invalid QoS level"):
            MQTTActionTransport(hostname="test-broker", qos=5)

    async def test_publish_action(self, mock_mqtt_client):
        """Test publishing an action to MQTT broker."""
        transport = MQTTActionTransport(
            hostname="test-broker", topic_prefix="test"
        )

        action = Action(
            ts=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            kind="test_action",
            stream_id="test_stream",
            payload={"result": "success"},
            ttl_ms=10000,
        )

        await transport.publish_action(action)

        # Verify publish was called
        mock_mqtt_client.publish.assert_called_once()

        # Verify call arguments
        call_args = mock_mqtt_client.publish.call_args
        assert call_args.kwargs["topic"] == "test/test_stream/actions"
        assert call_args.kwargs["qos"] == 1

        # Verify payload
        payload = json.loads(call_args.kwargs["payload"])
        assert payload["kind"] == "test_action"
        assert payload["stream_id"] == "test_stream"
        assert payload["payload"] == {"result": "success"}
        assert payload["ttl_ms"] == 10000

    async def test_subscribe_actions(self, mock_mqtt_client):
        """Test subscribing to actions from MQTT broker."""
        transport = MQTTActionTransport(
            hostname="test-broker", topic_prefix="test"
        )

        # Mock message
        mock_message = MagicMock()
        mock_message.topic.value = "test/stream1/actions"
        mock_message.payload = json.dumps(
            {
                "ts": "2025-01-15T10:30:00+00:00",
                "kind": "action_executed",
                "stream_id": "stream1",
                "payload": {"status": "complete"},
                "ttl_ms": 5000,
            }
        ).encode()

        async def mock_messages():
            yield mock_message

        mock_mqtt_client.messages = mock_messages()

        # Subscribe and collect actions
        actions = []
        async for action in transport.subscribe_actions():
            actions.append(action)
            break

        # Verify subscription
        mock_mqtt_client.subscribe.assert_called_once_with(
            "test/+/actions", qos=1
        )

        # Verify action
        assert len(actions) == 1
        assert actions[0].kind == "action_executed"
        assert actions[0].stream_id == "stream1"
        assert actions[0].payload == {"status": "complete"}
        assert actions[0].ttl_ms == 5000

    async def test_close(self, mock_mqtt_client):
        """Test closing MQTT connection."""
        transport = MQTTActionTransport(hostname="test-broker")

        # Create connection
        await transport._get_client()

        # Close
        await transport.close()

        mock_mqtt_client.__aexit__.assert_called_once()
        assert transport._client is None


class TestMQTTIntegration:
    """Integration tests for MQTT transports."""

    async def test_event_action_round_trip(self, mock_mqtt_client):
        """Test publishing event and action through MQTT."""
        event_transport = MQTTEventTransport(hostname="test-broker")
        action_transport = MQTTActionTransport(hostname="test-broker")

        # Create event
        event = Event(
            ts=datetime.now(timezone.utc),
            kind="user_signup",
            stream_id="users",
            payload={"user_id": "123"},
        )

        # Publish event
        await event_transport.publish_event(event)

        # Create action
        action = Action(
            ts=datetime.now(timezone.utc),
            kind="welcome_email",
            stream_id="users",
            payload={"user_id": "123", "email": "user@example.com"},
            ttl_ms=60000,
        )

        # Publish action
        await action_transport.publish_action(action)

        # Verify both were published
        assert mock_mqtt_client.publish.call_count == 2

    async def test_wildcard_subscription(self, mock_mqtt_client):
        """Test wildcard subscription pattern."""
        transport = MQTTEventTransport(
            hostname="test-broker", topic_prefix="production"
        )

        # Mock multiple stream messages
        messages = []
        for stream_id in ["stream1", "stream2", "stream3"]:
            msg = MagicMock()
            msg.topic.value = f"production/{stream_id}/events"
            msg.payload = json.dumps(
                {
                    "ts": "2025-01-15T10:30:00+00:00",
                    "kind": "event",
                    "stream_id": stream_id,
                    "payload": {},
                }
            ).encode()
            messages.append(msg)

        async def mock_messages():
            for msg in messages:
                yield msg

        mock_mqtt_client.messages = mock_messages()

        # Subscribe and collect
        events = []
        async for event in transport.subscribe_events():
            events.append(event)
            if len(events) >= 3:
                break

        # Should have received all events
        assert len(events) == 3
        stream_ids = {e.stream_id for e in events}
        assert stream_ids == {"stream1", "stream2", "stream3"}
