"""MQTT-based transport implementations for live-crew.

Provides event and action transports using MQTT protocol,
enabling integration with IoT devices and MQTT-based systems.
"""

import json
from typing import Any, AsyncIterator

from aiomqtt import Client, MqttError

from live_crew.core.models import Action, Event
from live_crew.interfaces.protocols import ActionTransport, EventTransport


class MQTTEventTransport(EventTransport):
    """MQTT-based event transport implementation.

    Publishes and subscribes to events using MQTT broker,
    suitable for IoT integration and edge computing scenarios.
    """

    def __init__(
        self,
        hostname: str,
        port: int = 1883,
        topic_prefix: str = "live",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
        client_id: str | None = None,
    ) -> None:
        """Initialize MQTT event transport.

        Args:
            hostname: MQTT broker hostname
            port: MQTT broker port (default: 1883)
            topic_prefix: Prefix for all MQTT topics (default: "live")
            username: Optional MQTT username for authentication
            password: Optional MQTT password for authentication
            qos: Quality of Service level 0, 1, or 2 (default: 1)
            client_id: Optional MQTT client ID

        Raises:
            ValueError: If QoS level is invalid
        """
        if qos not in (0, 1, 2):
            raise ValueError(f"Invalid QoS level: {qos}. Must be 0, 1, or 2")

        self.hostname = hostname
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self.qos = qos
        self.client_id = client_id
        self._client: Client | None = None

    async def _get_client(self) -> Client:
        """Get or create MQTT client.

        Returns:
            MQTT client instance

        Raises:
            MqttError: If connection to broker fails
        """
        if self._client is None:
            self._client = Client(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
                client_id=self.client_id,
            )
            await self._client.__aenter__()
        return self._client

    async def publish_event(self, event: Event[Any]) -> None:
        """Publish event to MQTT broker.

        Args:
            event: The event to publish

        Raises:
            MqttError: If publishing fails
        """
        client = await self._get_client()

        topic = f"{self.topic_prefix}/{event.stream_id}/events"
        payload = json.dumps(
            {
                "ts": event.ts.isoformat(),
                "kind": event.kind,
                "stream_id": event.stream_id,
                "payload": event.payload,
            }
        )

        await client.publish(
            topic=topic,
            payload=payload,
            qos=self.qos,
        )

    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        """Subscribe to events from MQTT broker.

        Yields:
            Events as they arrive from the broker

        Raises:
            MqttError: If subscription fails
            ValueError: If received message is invalid
        """
        client = await self._get_client()

        # Subscribe to all event streams using wildcard
        topic = f"{self.topic_prefix}/+/events"
        await client.subscribe(topic, qos=self.qos)

        async for message in client.messages:
            try:
                data = json.loads(message.payload.decode())
                event = Event[Any](**data)
                yield event

            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue processing other messages
                print(f"Invalid event message on {message.topic}: {e}")
                continue

    async def close(self) -> None:
        """Close MQTT connection gracefully.

        Should be called when transport is no longer needed.
        """
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None


class MQTTActionTransport(ActionTransport):
    """MQTT-based action transport implementation.

    Publishes and subscribes to actions using MQTT broker,
    suitable for IoT integration and edge computing scenarios.
    """

    def __init__(
        self,
        hostname: str,
        port: int = 1883,
        topic_prefix: str = "live",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
        client_id: str | None = None,
    ) -> None:
        """Initialize MQTT action transport.

        Args:
            hostname: MQTT broker hostname
            port: MQTT broker port (default: 1883)
            topic_prefix: Prefix for all MQTT topics (default: "live")
            username: Optional MQTT username for authentication
            password: Optional MQTT password for authentication
            qos: Quality of Service level 0, 1, or 2 (default: 1)
            client_id: Optional MQTT client ID

        Raises:
            ValueError: If QoS level is invalid
        """
        if qos not in (0, 1, 2):
            raise ValueError(f"Invalid QoS level: {qos}. Must be 0, 1, or 2")

        self.hostname = hostname
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self.qos = qos
        self.client_id = client_id
        self._client: Client | None = None

    async def _get_client(self) -> Client:
        """Get or create MQTT client.

        Returns:
            MQTT client instance

        Raises:
            MqttError: If connection to broker fails
        """
        if self._client is None:
            self._client = Client(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
                client_id=self.client_id,
            )
            await self._client.__aenter__()
        return self._client

    async def publish_action(self, action: Action[Any]) -> None:
        """Publish action to MQTT broker.

        Args:
            action: The action to publish

        Raises:
            MqttError: If publishing fails
        """
        client = await self._get_client()

        topic = f"{self.topic_prefix}/{action.stream_id}/actions"
        payload = json.dumps(
            {
                "ts": action.ts.isoformat(),
                "kind": action.kind,
                "stream_id": action.stream_id,
                "payload": action.payload,
                "ttl_ms": action.ttl_ms,
            }
        )

        await client.publish(
            topic=topic,
            payload=payload,
            qos=self.qos,
        )

    async def subscribe_actions(self) -> AsyncIterator[Action[Any]]:
        """Subscribe to actions from MQTT broker.

        Yields:
            Actions as they arrive from the broker

        Raises:
            MqttError: If subscription fails
            ValueError: If received message is invalid
        """
        client = await self._get_client()

        # Subscribe to all action streams using wildcard
        topic = f"{self.topic_prefix}/+/actions"
        await client.subscribe(topic, qos=self.qos)

        async for message in client.messages:
            try:
                data = json.loads(message.payload.decode())
                action = Action[Any](**data)
                yield action

            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue processing other messages
                print(f"Invalid action message on {message.topic}: {e}")
                continue

    async def close(self) -> None:
        """Close MQTT connection gracefully.

        Should be called when transport is no longer needed.
        """
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
