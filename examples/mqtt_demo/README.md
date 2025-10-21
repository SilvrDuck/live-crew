# MQTT Demo for live-crew

This example demonstrates using MQTT transports with live-crew for IoT and edge computing scenarios.

## Requirements

- MQTT broker (e.g., Mosquitto)
- Python 3.13+
- live-crew with dependencies installed

## Quick Start

### 1. Install MQTT Broker

**macOS:**
```bash
brew install mosquitto
brew services start mosquitto
```

**Ubuntu/Debian:**
```bash
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

**Docker:**
```bash
docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto
```

### 2. Install Dependencies

```bash
cd /path/to/live-crew
uv sync
```

### 3. Run the Demo

**Run all demos:**
```bash
python examples/mqtt_demo/mqtt_demo.py
```

**Run subscriber only (listens for 30 seconds):**
```bash
python examples/mqtt_demo/mqtt_demo.py --subscribe
```

**Run publisher only:**
```bash
python examples/mqtt_demo/mqtt_demo.py --publish
```

**Run in separate terminals:**
```bash
# Terminal 1 - Start subscriber
python examples/mqtt_demo/mqtt_demo.py --subscribe

# Terminal 2 - Publish events
python examples/mqtt_demo/mqtt_demo.py --publish
```

## What This Demo Shows

1. **Event Publishing**: How to publish events to MQTT broker
2. **Event Subscribing**: How to receive events from MQTT broker
3. **Action Publishing**: How to publish actions via MQTT
4. **Bidirectional Communication**: Event-action cycle for IoT scenarios

## MQTT Topics

The demo uses the following topic structure:

```
demo/+/events    # All sensor events
demo/+/actions   # All control actions
iot/+/events     # IoT-specific events
iot/+/actions    # IoT-specific actions
```

## Testing with MQTT CLI Tools

You can also test with Mosquitto command-line tools:

**Subscribe to events:**
```bash
mosquitto_sub -t "demo/+/events" -v
```

**Publish a test event:**
```bash
mosquitto_pub -t "demo/test/events" -m '{
  "ts": "2025-01-15T10:00:00Z",
  "kind": "test_event",
  "stream_id": "test",
  "payload": {"message": "Hello from MQTT"}
}'
```

## Next Steps

- Try different QoS levels (0, 1, 2)
- Connect to remote MQTT brokers
- Add authentication (username/password)
- Use TLS/SSL for secure connections
- Integrate with real IoT devices

## Troubleshooting

**Connection refused:**
- Check that Mosquitto is running: `brew services list` or `systemctl status mosquitto`
- Verify port 1883 is open: `netstat -an | grep 1883`

**No messages received:**
- Check topic patterns match
- Verify QoS levels are compatible
- Ensure subscriber is running before publisher

**Permission denied:**
- Check Mosquitto configuration allows anonymous connections
- Or configure authentication in the demo code
