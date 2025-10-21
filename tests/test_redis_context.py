"""Tests for Redis context backend implementation."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from live_crew.backends.redis_context import RedisContextBackend


@pytest.fixture
def mock_redis():
    """Create mock Redis client for testing."""
    mock_client = AsyncMock()

    # Setup default mock responses
    mock_client.hgetall.return_value = {}
    mock_client.hset.return_value = 1
    mock_client.delete.return_value = 1
    mock_client.scan.return_value = (0, [])
    mock_client.aclose = AsyncMock()

    return mock_client


@pytest.fixture
async def redis_backend():
    """Create Redis backend with mocked client."""
    backend = RedisContextBackend("redis://localhost:6379")
    return backend


class TestRedisContextBackendInit:
    """Test Redis backend initialization."""

    def test_initialization_with_default_url(self):
        """Test backend initializes with default Redis URL."""
        backend = RedisContextBackend()
        assert backend.redis_url == "redis://localhost:6379"
        assert backend._client is None

    def test_initialization_with_custom_url(self):
        """Test backend initializes with custom Redis URL."""
        custom_url = "redis://custom-host:6380"
        backend = RedisContextBackend(custom_url)
        assert backend.redis_url == custom_url
        assert backend._client is None


class TestRedisContextBackendKeyGeneration:
    """Test Redis key generation."""

    def test_make_key_format(self, redis_backend):
        """Test key generation follows correct format."""
        key = redis_backend._make_key("stream1", 0)
        assert key == "context:stream1:0"

    def test_make_key_with_different_indices(self, redis_backend):
        """Test key generation with various slice indices."""
        assert redis_backend._make_key("stream1", 0) == "context:stream1:0"
        assert redis_backend._make_key("stream1", 42) == "context:stream1:42"
        assert redis_backend._make_key("stream1", 999) == "context:stream1:999"

    def test_make_key_with_different_streams(self, redis_backend):
        """Test key generation with different stream IDs."""
        assert redis_backend._make_key("stream1", 0) == "context:stream1:0"
        assert redis_backend._make_key("stream2", 0) == "context:stream2:0"
        assert redis_backend._make_key("match42", 0) == "context:match42:0"


class TestRedisContextBackendGetSnapshot:
    """Test context snapshot retrieval."""

    @pytest.mark.asyncio
    async def test_get_snapshot_empty(self, redis_backend, mock_redis):
        """Test get_snapshot returns empty dict when no data exists."""
        redis_backend._client = mock_redis
        mock_redis.hgetall.return_value = {}

        result = await redis_backend.get_snapshot("stream1", 0)

        assert result == {}
        mock_redis.hgetall.assert_called_once_with("context:stream1:0")

    @pytest.mark.asyncio
    async def test_get_snapshot_with_data(self, redis_backend, mock_redis):
        """Test get_snapshot deserializes JSON data correctly."""
        redis_backend._client = mock_redis

        # Mock Redis response with bytes (as Redis returns)
        mock_redis.hgetall.return_value = {
            b"user_count": b'"42"',
            b"last_event": b'"signup"',
        }

        result = await redis_backend.get_snapshot("stream1", 0)

        assert result == {"user_count": "42", "last_event": "signup"}
        mock_redis.hgetall.assert_called_once_with("context:stream1:0")

    @pytest.mark.asyncio
    async def test_get_snapshot_with_complex_data(self, redis_backend, mock_redis):
        """Test get_snapshot handles complex nested data structures."""
        redis_backend._client = mock_redis

        # Mock complex nested structure
        complex_data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        mock_redis.hgetall.return_value = {b"data": json.dumps(complex_data).encode()}

        result = await redis_backend.get_snapshot("stream1", 0)

        assert result == {"data": complex_data}

    @pytest.mark.asyncio
    async def test_get_snapshot_creates_client_lazily(self, redis_backend):
        """Test get_snapshot creates Redis client on first use."""
        assert redis_backend._client is None

        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.hgetall.return_value = {}
            mock_from_url.return_value = mock_client

            await redis_backend.get_snapshot("stream1", 0)

            mock_from_url.assert_called_once()
            assert redis_backend._client is not None


class TestRedisContextBackendApplyDiff:
    """Test context diff application."""

    @pytest.mark.asyncio
    async def test_apply_diff_empty(self, redis_backend, mock_redis):
        """Test apply_diff with empty diff does nothing."""
        redis_backend._client = mock_redis

        await redis_backend.apply_diff("stream1", 0, {})

        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_diff_single_field(self, redis_backend, mock_redis):
        """Test apply_diff with single field update."""
        redis_backend._client = mock_redis

        await redis_backend.apply_diff("stream1", 0, {"counter": 1})

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == "context:stream1:0"
        assert call_args[1]["mapping"] == {"counter": "1"}

    @pytest.mark.asyncio
    async def test_apply_diff_multiple_fields(self, redis_backend, mock_redis):
        """Test apply_diff with multiple field updates."""
        redis_backend._client = mock_redis

        diff = {"user_count": 42, "last_event": "signup", "active": True}
        await redis_backend.apply_diff("stream1", 0, diff)

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        assert json.loads(mapping["user_count"]) == 42
        assert json.loads(mapping["last_event"]) == "signup"
        assert json.loads(mapping["active"]) is True

    @pytest.mark.asyncio
    async def test_apply_diff_complex_data(self, redis_backend, mock_redis):
        """Test apply_diff handles complex nested structures."""
        redis_backend._client = mock_redis

        diff = {
            "nested": {"key": "value", "num": 123},
            "list": [1, 2, 3],
            "null": None,
        }
        await redis_backend.apply_diff("stream1", 0, diff)

        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Verify JSON serialization/deserialization
        assert json.loads(mapping["nested"]) == {"key": "value", "num": 123}
        assert json.loads(mapping["list"]) == [1, 2, 3]
        assert json.loads(mapping["null"]) is None

    @pytest.mark.asyncio
    async def test_apply_diff_different_slices(self, redis_backend, mock_redis):
        """Test apply_diff with different slice indices."""
        redis_backend._client = mock_redis

        await redis_backend.apply_diff("stream1", 0, {"data": "slice0"})
        await redis_backend.apply_diff("stream1", 1, {"data": "slice1"})

        assert mock_redis.hset.call_count == 2
        calls = mock_redis.hset.call_args_list

        assert calls[0][0][0] == "context:stream1:0"
        assert calls[1][0][0] == "context:stream1:1"


class TestRedisContextBackendClearStream:
    """Test stream clearing functionality."""

    @pytest.mark.asyncio
    async def test_clear_stream_no_keys(self, redis_backend, mock_redis):
        """Test clear_stream when no keys exist."""
        redis_backend._client = mock_redis
        mock_redis.scan.return_value = (0, [])

        await redis_backend.clear_stream("stream1")

        mock_redis.scan.assert_called_once()
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_stream_single_batch(self, redis_backend, mock_redis):
        """Test clear_stream with keys in single batch."""
        redis_backend._client = mock_redis
        keys = [b"context:stream1:0", b"context:stream1:1"]
        mock_redis.scan.return_value = (0, keys)

        await redis_backend.clear_stream("stream1")

        mock_redis.scan.assert_called_once()
        mock_redis.delete.assert_called_once_with(*keys)

    @pytest.mark.asyncio
    async def test_clear_stream_multiple_batches(self, redis_backend, mock_redis):
        """Test clear_stream with keys across multiple SCAN iterations."""
        redis_backend._client = mock_redis

        keys1 = [b"context:stream1:0", b"context:stream1:1"]
        keys2 = [b"context:stream1:2", b"context:stream1:3"]

        # Mock SCAN to return data in two batches
        mock_redis.scan.side_effect = [(100, keys1), (0, keys2)]

        await redis_backend.clear_stream("stream1")

        assert mock_redis.scan.call_count == 2
        assert mock_redis.delete.call_count == 2
        mock_redis.delete.assert_any_call(*keys1)
        mock_redis.delete.assert_any_call(*keys2)

    @pytest.mark.asyncio
    async def test_clear_stream_pattern_matching(self, redis_backend, mock_redis):
        """Test clear_stream uses correct pattern matching."""
        redis_backend._client = mock_redis
        mock_redis.scan.return_value = (0, [])

        await redis_backend.clear_stream("stream1")

        # Verify scan was called with correct pattern
        call_args = mock_redis.scan.call_args
        assert call_args[1]["match"] == "context:stream1:*"


class TestRedisContextBackendClose:
    """Test connection cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_no_client(self, redis_backend):
        """Test close when no client was created."""
        assert redis_backend._client is None
        await redis_backend.close()
        assert redis_backend._client is None

    @pytest.mark.asyncio
    async def test_close_with_active_client(self, redis_backend, mock_redis):
        """Test close properly closes active client."""
        redis_backend._client = mock_redis

        await redis_backend.close()

        mock_redis.aclose.assert_called_once()
        assert redis_backend._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, redis_backend, mock_redis):
        """Test close can be called multiple times safely."""
        redis_backend._client = mock_redis

        await redis_backend.close()
        await redis_backend.close()

        # Should only close once
        mock_redis.aclose.assert_called_once()


class TestRedisContextBackendIntegration:
    """Integration-style tests with mock Redis."""

    @pytest.mark.asyncio
    async def test_full_context_lifecycle(self, redis_backend, mock_redis):
        """Test complete context lifecycle: apply, get, clear."""
        redis_backend._client = mock_redis

        # Apply initial diff
        await redis_backend.apply_diff("stream1", 0, {"counter": 1})

        # Mock get_snapshot response
        mock_redis.hgetall.return_value = {b"counter": b"1"}
        snapshot = await redis_backend.get_snapshot("stream1", 0)
        assert snapshot == {"counter": 1}

        # Clear stream
        mock_redis.scan.return_value = (0, [b"context:stream1:0"])
        await redis_backend.clear_stream("stream1")

        assert mock_redis.hset.call_count == 1
        assert mock_redis.hgetall.call_count == 1
        assert mock_redis.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_streams_isolation(self, redis_backend, mock_redis):
        """Test that different streams use different keys."""
        redis_backend._client = mock_redis

        await redis_backend.apply_diff("stream1", 0, {"data": "s1"})
        await redis_backend.apply_diff("stream2", 0, {"data": "s2"})

        calls = mock_redis.hset.call_args_list
        assert calls[0][0][0] == "context:stream1:0"
        assert calls[1][0][0] == "context:stream2:0"

    @pytest.mark.asyncio
    async def test_slice_progression(self, redis_backend, mock_redis):
        """Test context updates across multiple slices."""
        redis_backend._client = mock_redis

        # Simulate progression through slices
        for slice_idx in range(5):
            await redis_backend.apply_diff(
                "stream1", slice_idx, {"slice": slice_idx, "counter": slice_idx * 10}
            )

        # Should have 5 hset calls
        assert mock_redis.hset.call_count == 5

        # Verify keys for each slice
        calls = mock_redis.hset.call_args_list
        for i in range(5):
            assert calls[i][0][0] == f"context:stream1:{i}"


class TestRedisContextBackendErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self, redis_backend):
        """Test connection errors are propagated to caller."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await redis_backend.get_snapshot("stream1", 0)

    @pytest.mark.asyncio
    async def test_redis_operation_error_propagates(self, redis_backend, mock_redis):
        """Test Redis operation errors are propagated."""
        redis_backend._client = mock_redis
        mock_redis.hgetall.side_effect = Exception("Redis error")

        with pytest.raises(Exception, match="Redis error"):
            await redis_backend.get_snapshot("stream1", 0)
