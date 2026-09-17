from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis


class RateLimitExceeded(Exception):
    def __init__(self, reason: str, retry_after: int = 60) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


@dataclass(slots=True)
class RateLimitLease:
    limiter: RateLimiter
    user_id: str
    released: bool = False

    async def release(self) -> None:
        if not self.released:
            self.released = True
            await self.limiter.release(self.user_id)


class RateLimiter(Protocol):
    async def ready(self) -> None: ...

    async def acquire(self, user_id: str) -> RateLimitLease: ...

    async def release(self, user_id: str) -> None: ...

    async def close(self) -> None: ...


class MemoryRateLimiter:
    def __init__(self, requests_per_minute: int, max_concurrent: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._active: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def acquire(self, user_id: str) -> RateLimitLease:
        now = time.monotonic()
        async with self._lock:
            history = self._requests[user_id]
            while history and history[0] <= now - 60:
                history.popleft()
            if len(history) >= self.requests_per_minute:
                retry_after = max(1, int(60 - (now - history[0])))
                raise RateLimitExceeded("Per-minute request limit reached", retry_after)
            if self._active[user_id] >= self.max_concurrent:
                raise RateLimitExceeded("Concurrent request limit reached", 1)
            history.append(now)
            self._active[user_id] += 1
        return RateLimitLease(self, user_id)

    async def ready(self) -> None:
        return None

    async def release(self, user_id: str) -> None:
        async with self._lock:
            active = self._active.get(user_id, 0)
            if active <= 1:
                self._active.pop(user_id, None)
            else:
                self._active[user_id] = active - 1

    async def close(self) -> None:
        return None


ACQUIRE_SCRIPT = """
local rate = redis.call('INCR', KEYS[1])
if rate == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
if rate > tonumber(ARGV[2]) then
  return {-1, redis.call('TTL', KEYS[1])}
end
local active = redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[2], ARGV[4])
if active > tonumber(ARGV[3]) then
  redis.call('DECR', KEYS[2])
  return {-2, 1}
end
return {active, redis.call('TTL', KEYS[1])}
"""

RELEASE_SCRIPT = """
local active = tonumber(redis.call('GET', KEYS[1]) or '0')
if active <= 1 then
  redis.call('DEL', KEYS[1])
  return 0
end
return redis.call('DECR', KEYS[1])
"""


class RedisRateLimiter:
    def __init__(
        self,
        redis: Redis,
        requests_per_minute: int,
        max_concurrent: int,
        concurrency_ttl_seconds: int,
    ) -> None:
        self.redis = redis
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self.concurrency_ttl_seconds = concurrency_ttl_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:40]

    async def ready(self) -> None:
        await self.redis.ping()

    async def acquire(self, user_id: str) -> RateLimitLease:
        key = self._key(user_id)
        result = await self.redis.eval(
            ACQUIRE_SCRIPT,
            2,
            f"jjai:rate:{key}",
            f"jjai:active:{key}",
            60,
            self.requests_per_minute,
            self.max_concurrent,
            self.concurrency_ttl_seconds,
        )
        code = int(result[0])
        if code == -1:
            raise RateLimitExceeded("Per-minute request limit reached", max(1, int(result[1])))
        if code == -2:
            raise RateLimitExceeded("Concurrent request limit reached", 1)
        return RateLimitLease(self, user_id)

    async def release(self, user_id: str) -> None:
        key = self._key(user_id)
        await self.redis.eval(RELEASE_SCRIPT, 1, f"jjai:active:{key}")

    async def close(self) -> None:
        await self.redis.aclose()


def create_rate_limiter(
    *, redis_url: str, requests_per_minute: int, max_concurrent: int, request_timeout: float
) -> RateLimiter:
    if not redis_url:
        return MemoryRateLimiter(requests_per_minute, max_concurrent)
    redis = Redis.from_url(redis_url, decode_responses=True)
    return RedisRateLimiter(
        redis,
        requests_per_minute,
        max_concurrent,
        concurrency_ttl_seconds=max(60, int(request_timeout) + 60),
    )
