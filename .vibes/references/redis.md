# Redis Python Client Reference - 2024-2025 Best Practices

## Overview

Redis is an in-memory data structure store used as a database, cache, message broker, and streaming engine. The `redis-py` library provides async support through `redis.asyncio`.

## Installation

```bash
uv add redis
```

## Key Concepts

### Async Connection Management

```python
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

async def create_redis_client(url: str = "redis://localhost:6379") -> Redis:
    """Create async Redis client with connection pooling."""
    try:
        client = await redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,  # Auto-decode bytes to strings
            max_connections=10,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        # Test connection
        await client.ping()
        return client
    except ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        raise
    except TimeoutError as e:
        print(f"Redis connection timeout: {e}")
        raise

async def close_redis_client(client: Redis) -> None:
    """Properly close Redis connection."""
    await client.aclose()
```

### Hash Operations (Best for Context Storage)

```python
async def hash_operations(client: Redis, key: str):
    """Common hash operations for key-value storage."""
    
    # Set multiple fields at once
    await client.hset(
        key,
        mapping={
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
    )
    
    # Get single field
    value = await client.hget(key, "field1")
    
    # Get all fields and values
    all_data = await client.hgetall(key)
    # Returns: {"field1": "value1", "field2": "value2", "field3": "value3"}
    
    # Check if field exists
    exists = await client.hexists(key, "field1")
    
    # Delete specific fields
    await client.hdel(key, "field1", "field2")
    
    # Delete entire hash
    await client.delete(key)
```

### Pipeline Operations (Atomic Batching)

```python
async def pipeline_operations(client: Redis):
    """Use pipelines for atomic multi-operation transactions."""
    
    # Create pipeline (non-transactional by default)
    pipe = client.pipeline()
    
    # Queue multiple operations
    pipe.hset("context:stream1:0", "key1", "value1")
    pipe.hset("context:stream1:0", "key2", "value2")
    pipe.hgetall("context:stream1:0")
    
    # Execute all operations atomically
    results = await pipe.execute()
    # Returns list of results for each operation
    
    # Transactional pipeline (MULTI/EXEC)
    async with client.pipeline(transaction=True) as pipe:
        pipe.hset("counter", "value", "1")
        pipe.hincrby("counter", "value", 1)
        results = await pipe.execute()
```

### Key Management

```python
async def key_management(client: Redis):
    """Common key management operations."""
    
    # Set expiration (TTL)
    await client.expire("temporary:key", 3600)  # 1 hour
    
    # Set key with expiration at creation
    await client.setex("session:123", 1800, "session_data")  # 30 minutes
    
    # Check if key exists
    exists = await client.exists("mykey")
    
    # Get remaining TTL
    ttl = await client.ttl("mykey")
    # Returns -1 if no expiry, -2 if key doesn't exist
    
    # Delete keys
    await client.delete("key1", "key2", "key3")
    
    # Delete keys matching pattern
    cursor = 0
    while True:
        cursor, keys = await client.scan(cursor, match="context:stream1:*", count=100)
        if keys:
            await client.delete(*keys)
        if cursor == 0:
            break
```

## Context Backend Pattern for live-crew

### Basic Implementation

```python
import json
from typing import Any
import redis.asyncio as redis
from redis.asyncio import Redis

class RedisContextBackend:
    """Redis-based context storage with async support."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis context backend.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._client: Redis | None = None
    
    async def _get_client(self) -> Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Handle JSON encoding ourselves
                max_connections=10,
                socket_timeout=5,
            )
        return self._client
    
    def _make_key(self, stream_id: str, slice_idx: int) -> str:
        """Generate Redis key for context storage."""
        return f"context:{stream_id}:{slice_idx}"
    
    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        """Get context snapshot for a specific stream and slice."""
        client = await self._get_client()
        key = self._make_key(stream_id, slice_idx)
        
        # Get all hash fields
        data = await client.hgetall(key)
        
        # Decode and deserialize JSON values
        result = {}
        for field, value in data.items():
            field_str = field.decode() if isinstance(field, bytes) else field
            value_bytes = value if isinstance(value, bytes) else value.encode()
            result[field_str] = json.loads(value_bytes)
        
        return result
    
    async def apply_diff(
        self, stream_id: str, slice_idx: int, diff: dict[str, Any]
    ) -> None:
        """Apply a context diff for a specific stream and slice."""
        client = await self._get_client()
        key = self._make_key(stream_id, slice_idx)
        
        # Serialize values as JSON and store in hash
        mapping = {field: json.dumps(value) for field, value in diff.items()}
        
        if mapping:
            await client.hset(key, mapping=mapping)
    
    async def clear_stream(self, stream_id: str) -> None:
        """Clear all context data for a stream."""
        client = await self._get_client()
        pattern = f"context:{stream_id}:*"
        
        # Use SCAN to find and delete all matching keys
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

## Error Handling Patterns

### Retry with Exponential Backoff

```python
import asyncio
import random
from redis.exceptions import ConnectionError, TimeoutError

async def redis_operation_with_retry(
    operation_func,
    max_retries: int = 3,
    base_delay: float = 0.1
):
    """Execute Redis operation with retry logic."""
    
    for attempt in range(max_retries + 1):
        try:
            return await operation_func()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries:
                raise
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)
```

### Connection Pool Management

```python
class RedisConnectionPool:
    """Managed Redis connection pool."""
    
    def __init__(self, url: str, pool_size: int = 10):
        self.url = url
        self.pool_size = pool_size
        self._client: Redis | None = None
    
    async def get_client(self) -> Redis:
        """Get Redis client with connection pooling."""
        if self._client is None:
            self._client = await redis.from_url(
                self.url,
                max_connections=self.pool_size,
                encoding="utf-8",
                decode_responses=False,
            )
        return self._client
    
    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            client = await self.get_client()
            await client.ping()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close all connections in pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

## Testing Patterns

### Mock Redis for Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    
    # Setup mock responses
    mock_client.hgetall.return_value = {}
    mock_client.hset.return_value = 1
    mock_client.delete.return_value = 1
    mock_client.scan.return_value = (0, [])
    mock_client.ping.return_value = True
    
    return mock_client

async def test_redis_operations(mock_redis):
    """Test Redis operations with mocks."""
    backend = RedisContextBackend()
    backend._client = mock_redis
    
    # Test operations
    await backend.apply_diff("stream1", 0, {"key": "value"})
    
    # Verify calls
    mock_redis.hset.assert_called_once()
```

## Performance Considerations

### 1. Use Pipelining for Bulk Operations

```python
async def bulk_update_context(client: Redis, updates: list[tuple[str, dict]]):
    """Update multiple contexts efficiently using pipeline."""
    pipe = client.pipeline()
    
    for key, data in updates:
        mapping = {field: json.dumps(value) for field, value in data.items()}
        pipe.hset(key, mapping=mapping)
    
    await pipe.execute()
```

### 2. Optimize Key Scanning

```python
async def efficient_key_cleanup(client: Redis, pattern: str):
    """Efficiently delete keys matching pattern."""
    # Use larger count for better performance
    cursor = 0
    while True:
        cursor, keys = await client.scan(cursor, match=pattern, count=1000)
        if keys:
            # Delete in batches
            await client.delete(*keys)
        if cursor == 0:
            break
```

### 3. Connection Pooling

Always use connection pooling to avoid connection overhead:
- Set `max_connections` appropriate for your workload
- Reuse client instances across operations
- Close connections properly on shutdown

## Common Pitfalls

1. **Not Handling Bytes**: Redis returns bytes by default unless `decode_responses=True`
2. **Missing Async/Await**: All Redis operations are async and must be awaited
3. **Connection Leaks**: Always close connections with `await client.aclose()`
4. **No Error Handling**: Network operations can fail, always use try/except
5. **Inefficient Scanning**: Use SCAN with appropriate count, not KEYS in production
6. **JSON Serialization**: Store complex objects as JSON strings in hashes

## Best Practices for live-crew Integration

1. **Key Naming Convention**: Use hierarchical keys like `context:{stream_id}:{slice_idx}`
2. **Hash Storage**: Store context as Redis hashes for efficient field updates
3. **JSON Serialization**: Serialize complex values as JSON for type safety
4. **Pipeline Updates**: Use pipelines for atomic multi-field updates
5. **Connection Management**: Maintain single client instance with connection pooling
6. **Error Handling**: Implement retry logic for transient network errors
7. **Cleanup**: Use SCAN-based cleanup for pattern-matching key deletion
8. **Testing**: Use mocks for unit tests, real Redis for integration tests

## Version Notes

- **redis-py 5.0+**: Full async support with `redis.asyncio`
- **redis-py 4.x**: Legacy async support, upgrade recommended
- Requires Redis server 6.0+ for optimal performance
- Compatible with Redis Cluster and Sentinel for high availability
