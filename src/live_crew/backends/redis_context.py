"""Redis-based context backend implementation for live-crew.

Provides distributed context storage using Redis with async support.
Implements the ContextBackend protocol for seamless integration.
"""

import json
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis

from live_crew.interfaces.protocols import ContextBackend


class RedisContextBackend(ContextBackend):
    """Redis-based context backend with async support.

    Stores context in Redis using hash data structures for efficient
    field-level updates. Suitable for distributed scenarios and
    production deployments requiring persistent context storage.

    Args:
        redis_url: Redis connection URL (default: redis://localhost:6379)

    Example:
        >>> backend = RedisContextBackend("redis://localhost:6379")
        >>> await backend.apply_diff("stream1", 0, {"key": "value"})
        >>> snapshot = await backend.get_snapshot("stream1", 0)
        >>> await backend.close()
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        """Initialize Redis context backend.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._client: Redis | None = None

    async def _get_client(self) -> Redis:
        """Get or create Redis client with connection pooling.

        Returns:
            Redis client instance with connection pooling enabled
        """
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Handle JSON encoding ourselves
                max_connections=10,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        return self._client

    def _make_key(self, stream_id: str, slice_idx: int) -> str:
        """Generate Redis key for context storage.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index

        Returns:
            Redis key in format: context:{stream_id}:{slice_idx}
        """
        return f"context:{stream_id}:{slice_idx}"

    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        """Get context snapshot for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index

        Returns:
            Context snapshot as a dictionary (empty if not exists)

        Example:
            >>> snapshot = await backend.get_snapshot("stream1", 0)
            >>> print(snapshot)
            {"user_count": 42, "last_event": "signup"}
        """
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
        """Apply a context diff for a specific stream and slice.

        Args:
            stream_id: The stream identifier
            slice_idx: The slice index
            diff: The context diff to apply (key-value updates)

        Example:
            >>> await backend.apply_diff("stream1", 0, {
            ...     "user_count": 43,
            ...     "last_event": "login"
            ... })
        """
        client = await self._get_client()
        key = self._make_key(stream_id, slice_idx)

        # Serialize values as JSON and store in hash
        mapping = {field: json.dumps(value) for field, value in diff.items()}

        if mapping:
            await client.hset(key, mapping=mapping)

    async def clear_stream(self, stream_id: str) -> None:
        """Clear all context data for a stream.

        Uses SCAN to efficiently find and delete all keys matching
        the stream pattern without blocking Redis.

        Args:
            stream_id: The stream identifier to clear

        Example:
            >>> await backend.clear_stream("stream1")
        """
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
        """Close Redis connection and cleanup resources.

        Should be called when the backend is no longer needed to
        properly close connections and release resources.

        Example:
            >>> backend = RedisContextBackend()
            >>> # ... use backend ...
            >>> await backend.close()
        """
        if self._client:
            await self._client.aclose()
            self._client = None
