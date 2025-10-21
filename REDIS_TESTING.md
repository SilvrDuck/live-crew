# Redis Backend Testing Guide

## Quick Test with Docker

### 1. Start Redis
```bash
docker run -d -p 6379:6379 --name live-crew-redis redis:7-alpine
```

### 2. Run Example with Redis
```bash
cd examples/simple_multi_crew
export LIVE_CREW_KV_BACKEND=redis
python main.py
```

### 3. Verify Redis Storage
```bash
# Connect to Redis
redis-cli

# List all context keys
KEYS context:*

# View context for a specific stream/slice
HGETALL context:content_stream:0

# Watch real-time operations
MONITOR
```

### 4. Cleanup
```bash
# Stop and remove container
docker stop live-crew-redis
docker rm live-crew-redis
```

## Run Tests

### Prerequisites
```bash
# Install dependencies
uv sync --dev

# Or with pip
pip install redis pytest pytest-asyncio
```

### Run Redis Backend Tests
```bash
# All Redis tests (mock-based, no Redis server needed)
uv run -- pytest tests/test_redis_context.py -v

# Or with PYTHONPATH
PYTHONPATH=src pytest tests/test_redis_context.py -v
```

### Expected Test Output
```
tests/test_redis_context.py::TestRedisContextBackendInit::test_initialization_with_default_url PASSED
tests/test_redis_context.py::TestRedisContextBackendInit::test_initialization_with_custom_url PASSED
tests/test_redis_context.py::TestRedisContextBackendKeyGeneration::test_make_key_format PASSED
[... 37 more tests ...]

============================================ 40 passed in 0.5s ============================================
```

## Configuration Examples

### YAML Configuration
```yaml
# live-config.yaml
kv_backend: redis

vector:
  redis_url: redis://localhost:6379
```

### Environment Variables
```bash
export LIVE_CREW_KV_BACKEND=redis
# Redis URL defaults to redis://localhost:6379
```

### Programmatic Configuration
```python
from live_crew import Orchestrator
from live_crew.backends.redis_context import RedisContextBackend

# Option 1: Config file (auto-selects Redis)
orchestrator = Orchestrator.from_config("live-config.yaml")

# Option 2: Explicit backend
redis_backend = RedisContextBackend("redis://localhost:6379")
orchestrator = Orchestrator(context_backend=redis_backend)
```

## Troubleshooting

### Redis Connection Issues
```python
# Test Redis connection
import asyncio
import redis.asyncio as redis

async def test_connection():
    client = await redis.from_url("redis://localhost:6379")
    await client.ping()
    print("✓ Redis connection OK")
    await client.aclose()

asyncio.run(test_connection())
```

### Check Redis Server Status
```bash
# Check if Redis is running
docker ps | grep redis

# Check Redis logs
docker logs live-crew-redis

# Test connection with redis-cli
redis-cli ping
# Should return: PONG
```

## Integration Testing

### Test with Real Redis Server
```bash
# 1. Start Redis
docker run -d -p 6379:6379 --name test-redis redis:7-alpine

# 2. Run integration tests (if available)
uv run -- pytest tests/ -v -k redis

# 3. Manual verification
redis-cli
> KEYS *
> MONITOR
# (in another terminal)
python examples/simple_multi_crew/main.py

# 4. Cleanup
docker stop test-redis && docker rm test-redis
```

## Performance Testing

### Basic Benchmark
```python
import asyncio
import time
from live_crew.backends.redis_context import RedisContextBackend

async def benchmark():
    backend = RedisContextBackend()
    
    # Benchmark writes
    start = time.time()
    for i in range(1000):
        await backend.apply_diff("bench", i, {"counter": i, "data": "x" * 100})
    write_time = time.time() - start
    
    # Benchmark reads
    start = time.time()
    for i in range(1000):
        await backend.get_snapshot("bench", i)
    read_time = time.time() - start
    
    print(f"1000 writes: {write_time:.2f}s ({1000/write_time:.0f} ops/s)")
    print(f"1000 reads: {read_time:.2f}s ({1000/read_time:.0f} ops/s)")
    
    await backend.close()

asyncio.run(benchmark())
```

## Production Considerations

### Redis Configuration
```yaml
# production-redis.yaml
kv_backend: redis

vector:
  redis_url: redis://redis-cluster:6379
  # Or for Redis Sentinel:
  # redis_url: redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster
```

### Connection Pooling
The Redis backend uses connection pooling by default:
- Max connections: 10
- Socket timeout: 5 seconds
- Auto-reconnect enabled

### Monitoring
```bash
# Monitor Redis performance
redis-cli INFO stats
redis-cli INFO memory
redis-cli SLOWLOG GET 10
```

### Backup
```bash
# Create Redis backup
redis-cli SAVE

# Copy backup file
docker cp live-crew-redis:/data/dump.rdb ./backup/
```
