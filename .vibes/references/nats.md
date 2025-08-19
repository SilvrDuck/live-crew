# NATS.py Reference - 2024-2025 Best Practices

## Overview

NATS.py is the official Python client for NATS messaging with full async support and JetStream persistence. Updated for Python 3.8+ with modern async patterns.

## Installation

```bash
uv add nats-py
```

## Key Concepts

### Connection Management

```python
import asyncio
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError

async def connect_with_error_handling():
    try:
        nc = await nats.connect(
            servers=["nats://localhost:4222"],
            max_reconnect_attempts=5,
            reconnect_time_wait=2,
            ping_interval=120,
            max_outstanding_pings=2,
        )
        return nc
    except NoServersError:
        print("No servers available")
        raise
    except ConnectionClosedError:
        print("Connection was closed")
        raise

async def graceful_shutdown(nc):
    """Proper connection cleanup"""
    await nc.drain()  # Wait for pending messages
    await nc.close()
```

### JetStream Setup

```python
async def setup_jetstream(nc):
    # Create JetStream context
    js = nc.jetstream()

    # Create stream for persistence
    try:
        stream_info = await js.add_stream(
            name="events",
            subjects=["events.*"],
            storage="file",  # or "memory"
            max_msgs=1000000,
            max_bytes=1024*1024*1024,  # 1GB
            max_age=86400,  # 24 hours in seconds
            retention="limits",  # or "interest", "workqueue"
        )
    except Exception as e:
        # Stream might already exist
        stream_info = await js.stream_info("events")

    return js
```

## JetStream Patterns

### Publishing Messages

```python
async def publish_event(js, subject: str, data: bytes):
    """Publish with acknowledgment"""
    try:
        ack = await js.publish(
            subject=subject,
            payload=data,
            timeout=5.0,  # 5 second timeout
        )
        print(f"Published to {subject}, sequence: {ack.seq}")
        return ack
    except TimeoutError:
        print(f"Publish timeout for {subject}")
        raise
```

### Pull Subscriptions (Recommended for High-Throughput)

```python
async def consume_with_pull_subscription(js, stream_name: str, consumer_name: str):
    """Pull subscription with manual acknowledgment"""

    # Create pull consumer
    consumer_info = await js.add_consumer(
        stream=stream_name,
        config=nats.js.api.ConsumerConfig(
            durable_name=consumer_name,
            ack_policy="explicit",
            max_deliver=3,
            ack_wait=30,  # 30 seconds
            max_ack_pending=100,
        )
    )

    # Pull messages in batches
    psub = await js.pull_subscribe(
        subject="events.*",
        durable=consumer_name,
    )

    try:
        while True:
            # Fetch batch of messages
            msgs = await psub.fetch(
                batch=10,
                timeout=5.0,
            )

            for msg in msgs:
                try:
                    # Process message
                    await process_message(msg.data)

                    # Acknowledge successful processing
                    await msg.ack()

                except Exception as e:
                    print(f"Error processing message: {e}")
                    # Negative acknowledge to retry
                    await msg.nak()

    except asyncio.TimeoutError:
        # Normal timeout, continue
        pass
    except Exception as e:
        print(f"Consumer error: {e}")
        raise
```

### Push Subscriptions (Simpler but Less Control)

```python
async def consume_with_push_subscription(js, subject: str):
    """Push subscription with automatic delivery"""

    async def message_handler(msg):
        try:
            await process_message(msg.data)
            await msg.ack()
        except Exception as e:
            print(f"Handler error: {e}")
            await msg.nak()

    # Create push subscription
    psub = await js.subscribe(
        subject=subject,
        cb=message_handler,
        durable="push_consumer",
        config=nats.js.api.ConsumerConfig(
            ack_policy="explicit",
            max_deliver=3,
        ),
    )

    return psub
```

## KeyValue Store Best Practices

### Setup and Basic Operations

```python
async def setup_kv_store(js, bucket_name: str):
    """Create KV bucket with configuration"""
    try:
        kv = await js.create_key_value(
            bucket=bucket_name,
            history=5,  # Keep 5 versions (max 64)
            ttl=3600,   # 1 hour TTL
            max_value_size=1024*1024,  # 1MB max value
            storage="file",
        )
    except Exception:
        # Bucket might exist
        kv = js.key_value(bucket_name)

    return kv

async def kv_operations(kv):
    """Common KV operations"""

    # Put value
    await kv.put("user.123.name", b"John Doe")

    # Get value
    entry = await kv.get("user.123.name")
    if entry:
        print(f"Value: {entry.value.decode()}")
        print(f"Revision: {entry.revision}")

    # Delete key
    await kv.delete("user.123.name")

    # Conditional put (compare-and-swap)
    try:
        await kv.update("counter", b"1", last=0)
    except Exception as e:
        print(f"CAS failed: {e}")
```

### Watching for Changes

```python
async def watch_kv_changes(kv):
    """Watch for KV changes with patterns"""

    # Watch specific pattern
    watcher = await kv.watch("user.*")

    try:
        async for entry in watcher:
            if entry.operation == "PUT":
                print(f"Updated: {entry.key} = {entry.value}")
            elif entry.operation == "DELETE":
                print(f"Deleted: {entry.key}")
    except Exception as e:
        print(f"Watch error: {e}")
    finally:
        await watcher.stop()
```

## Async Patterns for Multi-Crew Orchestration

### Connection Pool Pattern

```python
class NATSConnectionPool:
    def __init__(self, servers: list[str], pool_size: int = 5):
        self.servers = servers
        self.pool_size = pool_size
        self._connections: list[nats.NATS] = []
        self._available = asyncio.Queue()

    async def initialize(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            nc = await nats.connect(servers=self.servers)
            self._connections.append(nc)
            await self._available.put(nc)

    async def get_connection(self) -> nats.NATS:
        """Get connection from pool"""
        return await self._available.get()

    async def return_connection(self, nc: nats.NATS):
        """Return connection to pool"""
        await self._available.put(nc)

    async def close_all(self):
        """Close all connections"""
        for nc in self._connections:
            await nc.close()
```

### Distributed Coordination Pattern

```python
async def coordinate_crews_with_nats(js, crew_configs: list[dict]):
    """Coordinate multiple crews using NATS subjects"""

    # Create coordination streams
    await js.add_stream(
        name="coordination",
        subjects=["crew.*.schedule", "crew.*.complete", "crew.*.heartbeat"],
        storage="memory",  # Fast access for coordination
        max_age=300,  # 5 minutes retention
    )

    # Schedule crews with dependencies
    tasks = []
    for config in crew_configs:
        task = asyncio.create_task(
            run_crew_with_coordination(js, config)
        )
        tasks.append(task)

    # Wait for all crews with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=300  # 5 minutes max
        )
    except asyncio.TimeoutError:
        print("Crew coordination timeout")
        # Cancel remaining tasks
        for task in tasks:
            task.cancel()
        raise

async def run_crew_with_coordination(js, config: dict):
    """Run single crew with NATS coordination"""
    crew_id = config["id"]

    # Publish schedule event
    await js.publish(
        f"crew.{crew_id}.schedule",
        b"starting"
    )

    try:
        # Check dependencies
        await wait_for_dependencies(js, config.get("dependencies", []))

        # Run crew logic
        await execute_crew_logic(config)

        # Signal completion
        await js.publish(
            f"crew.{crew_id}.complete",
            b"success"
        )

    except Exception as e:
        await js.publish(
            f"crew.{crew_id}.complete",
            f"error: {str(e)}".encode()
        )
        raise
```

## Error Handling Patterns

### Retry with Exponential Backoff

```python
import random

async def publish_with_retry(js, subject: str, data: bytes, max_retries: int = 3):
    """Publish with exponential backoff retry"""

    for attempt in range(max_retries + 1):
        try:
            return await js.publish(subject, data, timeout=5.0)

        except TimeoutError as e:
            if attempt == max_retries:
                raise

            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Publish failed, retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

        except ConnectionClosedError:
            # Don't retry connection errors
            raise
```

### Circuit Breaker Pattern

```python
class NATSCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, coro):
        """Execute coroutine with circuit breaker"""

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")

        try:
            result = await coro

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise
```

## Performance Considerations

### Message Batching

```python
async def batch_publisher(js, subject_prefix: str, batch_size: int = 100):
    """Batch messages for better throughput"""

    batch = []

    async def flush_batch():
        if not batch:
            return

        # Publish batch concurrently
        tasks = []
        for subject, data in batch:
            task = js.publish(subject, data)
            tasks.append(task)

        await asyncio.gather(*tasks)
        batch.clear()

    try:
        # Your message generation logic here
        for i in range(1000):
            subject = f"{subject_prefix}.{i % 10}"
            data = f"message_{i}".encode()

            batch.append((subject, data))

            if len(batch) >= batch_size:
                await flush_batch()

        # Flush remaining messages
        await flush_batch()

    except Exception as e:
        print(f"Batch publish error: {e}")
        raise
```

### Memory Management

```python
async def efficient_consumer(js, subject: str):
    """Memory-efficient consumer with flow control"""

    psub = await js.pull_subscribe(
        subject=subject,
        durable="efficient_consumer",
    )

    # Configure flow control
    await psub.consumer_info()

    try:
        while True:
            # Small batches to control memory usage
            msgs = await psub.fetch(batch=10, timeout=1.0)

            # Process messages sequentially to avoid memory spikes
            for msg in msgs:
                await process_message_efficiently(msg.data)
                await msg.ack()

                # Optional: add small delay to prevent CPU spikes
                await asyncio.sleep(0.001)

    except asyncio.TimeoutError:
        pass  # Normal timeout
```

## Testing Patterns

### Mock NATS for Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def mock_nats():
    """Mock NATS connection for testing"""

    mock_nc = AsyncMock()
    mock_js = AsyncMock()
    mock_kv = AsyncMock()

    mock_nc.jetstream.return_value = mock_js
    mock_js.key_value.return_value = mock_kv

    return mock_nc, mock_js, mock_kv

async def test_crew_coordination(mock_nats):
    """Test crew coordination logic"""
    mock_nc, mock_js, mock_kv = mock_nats

    # Setup mock responses
    mock_js.publish.return_value = MagicMock(seq=1)

    # Test your coordination logic
    await coordinate_crews_with_nats(mock_js, [{"id": "test_crew"}])

    # Verify calls
    mock_js.publish.assert_called()
```

## Common Pitfalls

1. **Connection Leaks**: Always use `await nc.close()` or `await nc.drain()`
2. **Message Acknowledgment**: Don't forget to ack/nak messages in pull subscriptions
3. **Stream Limits**: Configure appropriate limits to prevent unbounded growth
4. **Timeout Handling**: Set reasonable timeouts for all operations
5. **Error Recovery**: Implement proper retry and circuit breaker patterns
6. **Memory Usage**: Use pull subscriptions with small batches for high-throughput scenarios

## 2024-2025 Updates

- Full Python 3.8+ support with modern async patterns
- Enhanced JetStream KV store with history and TTL support
- Improved error handling with specific exception types
- Better connection management and reconnection logic
- Support for ordered consumers and flow control
- Enhanced security with TLS and NKEYS authentication
