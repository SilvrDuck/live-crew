#!/usr/bin/env python3
"""MQTT Transport Demo for live-crew.

This example demonstrates how to:
1. Publish events to an MQTT broker
2. Subscribe to events from MQTT
3. Use MQTT for IoT and edge computing scenarios

Requirements:
- MQTT broker running (e.g., Mosquitto on localhost:1883)
- Install: `uv sync` to get aiomqtt dependency

Quick start with Mosquitto:
  # macOS: brew install mosquitto && brew services start mosquitto
  # Ubuntu: sudo apt install mosquitto && sudo systemctl start mosquitto
  # Docker: docker run -d -p 1883:1883 eclipse-mosquitto
"""

import asyncio
from datetime import datetime, timezone

from live_crew import Event, Action, MQTTEventTransport, MQTTActionTransport


async def publisher_demo():
    """Demonstrate publishing events to MQTT."""
    print("=" * 60)
    print("MQTT Event Publisher Demo")
    print("=" * 60)

    transport = MQTTEventTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="demo",
        qos=1,
    )

    print("\n📤 Publishing sensor events to MQTT...")

    # Simulate IoT sensor events
    sensor_events = [
        {
            "kind": "temperature_reading",
            "stream_id": "sensor_01",
            "payload": {"temperature": 22.5, "unit": "celsius", "location": "room_a"},
        },
        {
            "kind": "humidity_reading",
            "stream_id": "sensor_01",
            "payload": {"humidity": 65, "unit": "percent", "location": "room_a"},
        },
        {
            "kind": "motion_detected",
            "stream_id": "sensor_02",
            "payload": {"motion": True, "confidence": 0.95, "location": "hallway"},
        },
    ]

    for event_data in sensor_events:
        event = Event(
            ts=datetime.now(timezone.utc),
            kind=event_data["kind"],
            stream_id=event_data["stream_id"],
            payload=event_data["payload"],
        )

        await transport.publish_event(event)
        print(f"  ✓ Published: {event.kind} from {event.stream_id}")
        await asyncio.sleep(0.5)

    await transport.close()
    print("\n✅ Publisher complete")


async def subscriber_demo(duration: int = 10):
    """Demonstrate subscribing to events from MQTT."""
    print("=" * 60)
    print("MQTT Event Subscriber Demo")
    print("=" * 60)

    transport = MQTTEventTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="demo",
        qos=1,
    )

    print(f"\n📥 Listening for events on MQTT (for {duration} seconds)...")
    print("   Topic pattern: demo/+/events")
    print("\nWaiting for events...\n")

    try:
        # Set timeout for demo
        async def listen():
            async for event in transport.subscribe_events():
                print(f"📨 Received Event:")
                print(f"   Kind: {event.kind}")
                print(f"   Stream: {event.stream_id}")
                print(f"   Payload: {event.payload}")
                print(f"   Timestamp: {event.ts}")
                print()

        await asyncio.wait_for(listen(), timeout=duration)

    except asyncio.TimeoutError:
        print(f"⏱️  {duration} second demo period complete")

    finally:
        await transport.close()
        print("✅ Subscriber closed")


async def action_demo():
    """Demonstrate publishing actions to MQTT."""
    print("=" * 60)
    print("MQTT Action Publisher Demo")
    print("=" * 60)

    transport = MQTTActionTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="demo",
        qos=1,
    )

    print("\n📤 Publishing control actions to MQTT...")

    # Simulate control actions
    actions = [
        {
            "kind": "turn_on_lights",
            "stream_id": "actuator_01",
            "payload": {"device": "lights_room_a", "brightness": 80},
            "ttl_ms": 5000,
        },
        {
            "kind": "adjust_thermostat",
            "stream_id": "actuator_02",
            "payload": {"device": "thermostat", "temperature": 21, "mode": "heat"},
            "ttl_ms": 10000,
        },
    ]

    for action_data in actions:
        action = Action(
            ts=datetime.now(timezone.utc),
            kind=action_data["kind"],
            stream_id=action_data["stream_id"],
            payload=action_data["payload"],
            ttl_ms=action_data["ttl_ms"],
        )

        await transport.publish_action(action)
        print(f"  ✓ Published: {action.kind} to {action.stream_id}")
        print(f"    TTL: {action.ttl_ms}ms")
        await asyncio.sleep(0.5)

    await transport.close()
    print("\n✅ Action publisher complete")


async def bidirectional_demo():
    """Demonstrate bidirectional MQTT communication."""
    print("=" * 60)
    print("MQTT Bidirectional Demo")
    print("=" * 60)

    event_transport = MQTTEventTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="iot",
        qos=1,
        client_id="demo_event_client",
    )

    action_transport = MQTTActionTransport(
        hostname="localhost",
        port=1883,
        topic_prefix="iot",
        qos=1,
        client_id="demo_action_client",
    )

    print("\n🔄 Simulating IoT event-action cycle...")

    # Publish sensor event
    sensor_event = Event(
        ts=datetime.now(timezone.utc),
        kind="temperature_alert",
        stream_id="sensors",
        payload={"temperature": 28, "threshold": 25, "alert": "high"},
    )

    print(f"\n📤 Sensor Event: {sensor_event.kind}")
    print(f"   Payload: {sensor_event.payload}")
    await event_transport.publish_event(sensor_event)

    # Simulate processing and generate action
    await asyncio.sleep(0.2)

    control_action = Action(
        ts=datetime.now(timezone.utc),
        kind="activate_cooling",
        stream_id="sensors",
        payload={"device": "ac_unit", "mode": "cool", "target": 23},
        ttl_ms=60000,
    )

    print(f"\n📤 Control Action: {control_action.kind}")
    print(f"   Payload: {control_action.payload}")
    await action_transport.publish_action(control_action)

    await event_transport.close()
    await action_transport.close()

    print("\n✅ Bidirectional demo complete")


async def main():
    """Run all MQTT demos."""
    print("\n🚀 live-crew MQTT Transport Demos\n")

    # Demo 1: Publisher
    await publisher_demo()
    print()

    # Demo 2: Actions
    await action_demo()
    print()

    # Demo 3: Bidirectional
    await bidirectional_demo()
    print()

    print("=" * 60)
    print("💡 To run the subscriber demo separately:")
    print("   python mqtt_demo.py --subscribe")
    print()
    print("💡 To publish and subscribe simultaneously:")
    print("   Terminal 1: python mqtt_demo.py --subscribe")
    print("   Terminal 2: python mqtt_demo.py --publish")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--subscribe" in sys.argv:
        # Run subscriber for 30 seconds
        asyncio.run(subscriber_demo(duration=30))
    elif "--publish" in sys.argv:
        # Run publisher only
        asyncio.run(publisher_demo())
    else:
        # Run all demos
        asyncio.run(main())
