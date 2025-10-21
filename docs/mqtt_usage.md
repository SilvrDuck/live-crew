# MQTT Transport Usage Examples

This guide demonstrates how to use MQTT transports with live-crew for IoT integration and edge computing scenarios.

## Installation

The MQTT transport requires the `aiomqtt` library, which is included in live-crew's dependencies:

```bash
uv sync
```

## Basic Usage

### Publishing Events to MQTT

```python
import asyncio
from datetime import datetime, timezone
from live_crew import Event, MQTTEventTransport

async def publish_events():
    """Publish events to MQTT broker."""
    # Create MQTT event transport
    transport = MQTTEventTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="live",
        qos=1,
    )
    
    # Create and publish event
    event = Event(
        ts=datetime.now(timezone.utc),
        kind="sensor_reading",
        stream_id="sensors",
        payload={"temperature": 22.5, "humidity": 65},
    )
    
    await transport.publish_event(event)
    print("Event published to MQTT")
    
    # Clean up
    await transport.close()

asyncio.run(publish_events())
```

### Subscribing to Events from MQTT

```python
import asyncio
from live_crew import MQTTEventTransport

async def subscribe_events():
    """Subscribe to events from MQTT broker."""
    transport = MQTTEventTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="live",
    )
    
    print("Listening for events on MQTT...")
    
    try:
        async for event in transport.subscribe_events():
            print(f"Received event: {event.kind}")
            print(f"  Stream: {event.stream_id}")
            print(f"  Payload: {event.payload}")
            print(f"  Timestamp: {event.ts}")
    finally:
        await transport.close()

asyncio.run(subscribe_events())
```

## Advanced Configuration

### Using MQTT Configuration Model

```python
from live_crew import MQTTConfig, MQTTEventTransport

# Define configuration
config = MQTTConfig(
    hostname="mqtt.example.com",
    port=1883,
    username="live-crew",
    password="secure-password",
    topic_prefix="production/live",
    qos=2,
)

# Create transport from config
transport = MQTTEventTransport(
    hostname=config.hostname,
    port=config.port,
    username=config.username,
    password=config.password,
    topic_prefix=config.topic_prefix,
    qos=config.qos,
)
```

## MQTT Topic Structure

live-crew uses a hierarchical topic structure for MQTT:

```
{topic_prefix}/{stream_id}/events    # Event input
{topic_prefix}/{stream_id}/actions   # Action output
```

### Examples

- Default: `live/sensors/events`, `live/sensors/actions`
- Custom: `production/live/stream1/events`, `production/live/stream1/actions`
- IoT: `iot/devices/temp_sensor_01/events`

## Best Practices

1. **Use QoS 1 for most cases**: Balances reliability and performance
2. **Set meaningful client IDs**: Helps debugging and monitoring
3. **Use authentication**: Always secure production brokers
4. **Clean up connections**: Always call `transport.close()`
5. **Topic naming**: Use hierarchical structure for organization

## References

- [MQTT Protocol Specification](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)
- [aiomqtt Documentation](https://sbtinstruments.github.io/aiomqtt/)
