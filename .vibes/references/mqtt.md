# MQTT Reference - 2024-2025 Best Practices

## Overview

MQTT (Message Queuing Telemetry Transport) is a lightweight publish-subscribe messaging protocol designed for constrained devices and low-bandwidth, high-latency networks. It's ideal for IoT applications and real-time event streaming.

## Installation

For async Python applications (recommended for live-crew):

```bash
uv add aiomqtt
```

Alternative (sync with threading):
```bash
uv add paho-mqtt
```

## Library Selection: aiomqtt vs paho-mqtt

### aiomqtt (Recommended for live-crew)
- **Pros**: Native async/await support, integrates seamlessly with asyncio
- **Pros**: Built on top of paho-mqtt, inherits its stability
- **Pros**: Clean API designed for modern Python 3.8+
- **Cons**: Newer library, smaller community than paho-mqtt

### paho-mqtt
- **Pros**: Official Eclipse Foundation library, very mature
- **Pros**: Large community, extensive documentation
- **Cons**: Primarily synchronous, requires threading for async patterns
- **Cons**: Less Pythonic API for async use cases

**Decision for live-crew**: Use `aiomqtt` for native async support matching our NATS and asyncio architecture.

## Key Concepts

### MQTT Protocol Basics

- **Broker**: Central server that routes messages (e.g., Mosquitto, HiveMQ, EMQX)
- **Topics**: Hierarchical namespaces for messages (e.g., `live/match42/events`)
- **QoS Levels**:
  - **0**: At most once (fire and forget)
  - **1**: At least once (acknowledged delivery)
  - **2**: Exactly once (guaranteed delivery)
- **Retained Messages**: Last message on topic persists for new subscribers
- **Clean Session**: Whether to persist session state across connections

### Topic Structure for live-crew

Following NATS subject patterns from spec:

```
live/<stream_id>/events          # Event input stream
live/<stream_id>/actions         # Action output stream
live/<stream_id>/errors          # Error messages
live/<stream_id>/heartbeats      # Liveness signals
live/<stream_id>/ctx             # Context updates
```

## Basic Usage with aiomqtt

### Connection and Publishing

```python
import asyncio
from aiomqtt import Client, MqttError
import json

async def publish_event():
    """Basic MQTT publishing pattern"""
    try:
        async with Client(
            hostname="localhost",
            port=1883,
            username="user",
            password="pass",
            client_id="live-crew-publisher",
            clean_session=True,
        ) as client:
            # Publish with QoS 1 (at least once delivery)
            await client.publish(
                topic="live/match42/events",
                payload=json.dumps({"kind": "goal_scored", "team": "home"}),
                qos=1,
                retain=False,
            )
            print("Event published successfully")
            
    except MqttError as e:
        print(f"MQTT error: {e}")
        raise
```

### Subscribing to Messages

```python
async def subscribe_events():
    """Basic MQTT subscription pattern"""
    async with Client(
        hostname="localhost",
        port=1883,
        client_id="live-crew-subscriber",
    ) as client:
        # Subscribe to topic with QoS 1
        await client.subscribe("live/+/events", qos=1)
        
        # Process messages as they arrive
        async for message in client.messages:
            try:
                payload = json.loads(message.payload.decode())
                print(f"Received on {message.topic}: {payload}")
                
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
```

## Advanced Patterns for live-crew

### Reconnection Handling

```python
import asyncio
from aiomqtt import Client, MqttError

async def connect_with_retry(
    hostname: str,
    port: int = 1883,
    max_retries: int = 5,
    retry_delay: float = 2.0,
):
    """Connect with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            client = Client(
                hostname=hostname,
                port=port,
                keepalive=60,
                timeout=10,
            )
            await client.__aenter__()
            return client
            
        except MqttError as e:
            if attempt == max_retries - 1:
                raise
            
            delay = retry_delay * (2 ** attempt)
            print(f"Connection failed, retrying in {delay}s...")
            await asyncio.sleep(delay)
```

### Message Queue with Buffering

```python
import asyncio
from collections import deque
from aiomqtt import Client

class MQTTMessageQueue:
    """Buffered message queue for high-throughput scenarios"""
    
    def __init__(self, max_size: int = 1000):
        self.queue: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def publish_batch(self, client: Client, topic: str):
        """Publish messages in batches for better throughput"""
        async with self._lock:
            batch = []
            while self.queue and len(batch) < 100:
                batch.append(self.queue.popleft())
            
            for message in batch:
                await client.publish(
                    topic=topic,
                    payload=message,
                    qos=1,
                )
    
    async def add(self, message: bytes):
        """Add message to queue"""
        async with self._lock:
            self.queue.append(message)
```

### Wildcard Subscriptions

```python
async def subscribe_multiple_streams():
    """Subscribe to multiple streams with wildcards"""
    async with Client("localhost") as client:
        # Single-level wildcard (+)
        await client.subscribe("live/+/events", qos=1)
        
        # Multi-level wildcard (#)
        await client.subscribe("live/match42/#", qos=1)
        
        async for message in client.messages:
            # Parse topic to determine stream
            topic_parts = message.topic.value.split("/")
            stream_id = topic_parts[1] if len(topic_parts) > 1 else "unknown"
            print(f"Stream {stream_id}: {message.payload}")
```

## Integration with live-crew Architecture

### Event Transport Implementation

```python
from typing import Any, AsyncIterator
from aiomqtt import Client, MqttError
import json
from datetime import datetime, timezone

from live_crew.core.models import Event
from live_crew.interfaces.protocols import EventTransport


class MQTTEventTransport(EventTransport):
    """MQTT-based event transport for live-crew."""
    
    def __init__(
        self,
        hostname: str,
        port: int = 1883,
        topic_prefix: str = "live",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
    ):
        self.hostname = hostname
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self.qos = qos
        self._client: Client | None = None
    
    async def _get_client(self) -> Client:
        """Get or create MQTT client"""
        if self._client is None:
            self._client = Client(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
            )
            await self._client.__aenter__()
        return self._client
    
    async def publish_event(self, event: Event[Any]) -> None:
        """Publish event to MQTT broker"""
        client = await self._get_client()
        
        topic = f"{self.topic_prefix}/{event.stream_id}/events"
        payload = json.dumps({
            "ts": event.ts.isoformat(),
            "kind": event.kind,
            "stream_id": event.stream_id,
            "payload": event.payload,
        })
        
        await client.publish(
            topic=topic,
            payload=payload,
            qos=self.qos,
        )
    
    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        """Subscribe to events from MQTT broker"""
        client = await self._get_client()
        
        # Subscribe to all event streams
        topic = f"{self.topic_prefix}/+/events"
        await client.subscribe(topic, qos=self.qos)
        
        async for message in client.messages:
            try:
                data = json.loads(message.payload.decode())
                event = Event[Any](**data)
                yield event
                
            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue processing
                print(f"Invalid event message: {e}")
                continue
    
    async def close(self):
        """Close MQTT connection"""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
```

### Action Transport Implementation

```python
from live_crew.core.models import Action
from live_crew.interfaces.protocols import ActionTransport


class MQTTActionTransport(ActionTransport):
    """MQTT-based action transport for live-crew."""
    
    def __init__(
        self,
        hostname: str,
        port: int = 1883,
        topic_prefix: str = "live",
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
    ):
        self.hostname = hostname
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self.qos = qos
        self._client: Client | None = None
    
    async def _get_client(self) -> Client:
        """Get or create MQTT client"""
        if self._client is None:
            self._client = Client(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
            )
            await self._client.__aenter__()
        return self._client
    
    async def publish_action(self, action: Action[Any]) -> None:
        """Publish action to MQTT broker"""
        client = await self._get_client()
        
        topic = f"{self.topic_prefix}/{action.stream_id}/actions"
        payload = json.dumps({
            "ts": action.ts.isoformat(),
            "kind": action.kind,
            "stream_id": action.stream_id,
            "payload": action.payload,
            "ttl_ms": action.ttl_ms,
        })
        
        await client.publish(
            topic=topic,
            payload=payload,
            qos=self.qos,
        )
    
    async def subscribe_actions(self) -> AsyncIterator[Action[Any]]:
        """Subscribe to actions from MQTT broker"""
        client = await self._get_client()
        
        # Subscribe to all action streams
        topic = f"{self.topic_prefix}/+/actions"
        await client.subscribe(topic, qos=self.qos)
        
        async for message in client.messages:
            try:
                data = json.loads(message.payload.decode())
                action = Action[Any](**data)
                yield action
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Invalid action message: {e}")
                continue
    
    async def close(self):
        """Close MQTT connection"""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
```

## Configuration Model

```python
from pydantic import BaseModel, Field


class MQTTConfig(BaseModel):
    """MQTT connection configuration."""
    
    hostname: str = Field(
        default="localhost",
        description="MQTT broker hostname"
    )
    port: int = Field(
        default=1883,
        ge=1,
        le=65535,
        description="MQTT broker port"
    )
    username: str | None = Field(
        default=None,
        description="MQTT username for authentication"
    )
    password: str | None = Field(
        default=None,
        description="MQTT password for authentication"
    )
    topic_prefix: str = Field(
        default="live",
        pattern=r"^[a-zA-Z0-9_/-]+$",
        description="Topic prefix for all MQTT topics"
    )
    qos: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Quality of Service level (0, 1, or 2)"
    )
    keepalive: int = Field(
        default=60,
        ge=1,
        description="Keepalive interval in seconds"
    )
    timeout: int = Field(
        default=10,
        ge=1,
        description="Connection timeout in seconds"
    )
```

## Testing Patterns

### Mock MQTT for Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
async def mock_mqtt_client():
    """Mock MQTT client for testing"""
    with patch('aiomqtt.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock context manager
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        yield mock_client


async def test_mqtt_event_transport_publish(mock_mqtt_client):
    """Test MQTT event publishing"""
    from live_crew.transports.mqtt import MQTTEventTransport
    from live_crew.core.models import Event
    from datetime import datetime, timezone
    
    transport = MQTTEventTransport(hostname="test-broker")
    
    event = Event(
        ts=datetime.now(timezone.utc),
        kind="test_event",
        stream_id="test_stream",
        payload={"data": "test"},
    )
    
    await transport.publish_event(event)
    
    # Verify publish was called
    mock_mqtt_client.publish.assert_called_once()
    call_args = mock_mqtt_client.publish.call_args
    
    assert "test_stream" in call_args.kwargs["topic"]
    assert "test_event" in call_args.kwargs["payload"]
```

### Integration Tests with Test Broker

For integration testing, use a containerized MQTT broker:

```python
import pytest
from testcontainers.core.container import DockerContainer


@pytest.fixture(scope="session")
async def mqtt_broker():
    """Start Mosquitto MQTT broker for integration tests"""
    with DockerContainer("eclipse-mosquitto:2") \
        .with_exposed_ports(1883) \
        .with_command("mosquitto -c /mosquitto-no-auth.conf") as container:
        
        # Wait for broker to be ready
        import asyncio
        await asyncio.sleep(2)
        
        host = container.get_container_host_ip()
        port = container.get_exposed_port(1883)
        
        yield {"hostname": host, "port": int(port)}
```

## Performance Considerations

### QoS Level Selection

- **QoS 0**: Best for high-frequency, low-importance data (metrics, heartbeats)
- **QoS 1**: Recommended for events and actions (balances reliability and performance)
- **QoS 2**: Only for critical data requiring exactly-once delivery (avoid for high throughput)

### Connection Pooling

For high-throughput scenarios, maintain persistent connections rather than reconnecting:

```python
class MQTTConnectionPool:
    """Connection pool for MQTT clients"""
    
    def __init__(self, hostname: str, pool_size: int = 5):
        self.hostname = hostname
        self.pool_size = pool_size
        self._clients: list[Client] = []
        self._available = asyncio.Queue()
    
    async def initialize(self):
        """Initialize connection pool"""
        for i in range(self.pool_size):
            client = Client(
                hostname=self.hostname,
                client_id=f"live-crew-{i}",
            )
            await client.__aenter__()
            self._clients.append(client)
            await self._available.put(client)
    
    async def get_client(self) -> Client:
        """Get client from pool"""
        return await self._available.get()
    
    async def return_client(self, client: Client):
        """Return client to pool"""
        await self._available.put(client)
```

## Security Best Practices

### TLS/SSL Encryption

```python
import ssl

async def create_secure_client():
    """Create MQTT client with TLS"""
    tls_context = ssl.create_default_context()
    tls_context.check_hostname = True
    tls_context.verify_mode = ssl.CERT_REQUIRED
    
    client = Client(
        hostname="mqtt.example.com",
        port=8883,  # Standard MQTT TLS port
        tls_context=tls_context,
        username="user",
        password="pass",
    )
    return client
```

### Authentication

Always use username/password authentication in production:

```python
# From environment variables
import os

client = Client(
    hostname=os.getenv("MQTT_HOSTNAME"),
    port=int(os.getenv("MQTT_PORT", "1883")),
    username=os.getenv("MQTT_USERNAME"),
    password=os.getenv("MQTT_PASSWORD"),
)
```

## Common Pitfalls

1. **Connection Leaks**: Always use async context managers or explicit cleanup
2. **QoS Misuse**: QoS 2 can severely impact performance - use sparingly
3. **Topic Naming**: Avoid special characters (`+`, `#`, `$`) in custom topics
4. **Message Size**: MQTT has broker-dependent size limits (typically 256MB max)
5. **Retained Messages**: Be careful with retained messages - they persist until deleted
6. **Clean Session**: Understand session persistence implications for subscribers

## MQTT vs NATS for live-crew

| Aspect | MQTT | NATS (current) |
|--------|------|----------------|
| **Use Case** | IoT, constrained devices | Microservices, cloud-native |
| **Performance** | Moderate throughput | Very high throughput |
| **Persistence** | Via broker | JetStream add-on |
| **QoS Levels** | 0, 1, 2 | Core, JetStream |
| **Topic Model** | Hierarchical | Subject-based |
| **Ecosystem** | Mature, widely adopted | Modern, growing |

**Recommendation**: MQTT is excellent for edge devices and IoT integration, while NATS excels at cloud-native orchestration. Use MQTT as an alternative transport for edge scenarios.

## References

- [aiomqtt Documentation](https://sbtinstruments.github.io/aiomqtt/)
- [MQTT Specification 5.0](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)
- [Eclipse Paho Python](https://eclipse.dev/paho/index.php?page=clients/python/index.php)
- [MQTT Essentials](https://www.hivemq.com/mqtt-essentials/)
