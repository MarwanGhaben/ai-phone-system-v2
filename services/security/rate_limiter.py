import time

import redis.asyncio as redis

from config.settings import get_settings


_INCREMENT_WITH_EXPIRY = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('TTL', KEYS[1])}
"""


def create_redis_client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


class RedisRateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
    ) -> None:
        self._redis = redis_client
        self._requests_per_minute = requests_per_minute
        self._burst_limit = burst_limit

    async def check(self, client_ip: str, now: float | None = None) -> tuple[bool, dict]:
        timestamp = time.time() if now is None else now
        burst_count, burst_ttl = await self._increment(
            f"rate-limit:burst:{client_ip}:{int(timestamp)}", 2
        )
        minute_count, minute_ttl = await self._increment(
            f"rate-limit:minute:{client_ip}:{int(timestamp // 60)}", 61
        )

        if burst_count > self._burst_limit:
            return True, self._details("burst_limit_exceeded", self._burst_limit, burst_ttl)
        if minute_count > self._requests_per_minute:
            return True, self._details(
                "rate_limit_exceeded", self._requests_per_minute, minute_ttl
            )
        return False, {}

    async def _increment(self, key: str, ttl: int) -> tuple[int, int]:
        count, remaining = await self._redis.eval(
            _INCREMENT_WITH_EXPIRY, 1, key, ttl
        )
        return int(count), max(1, int(remaining))

    @staticmethod
    def _details(reason: str, limit: int, retry_after: int) -> dict:
        return {"reason": reason, "limit": limit, "retry_after": retry_after}


class LoginAttemptLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        max_attempts: int = 5,
        lockout_seconds: int = 900,
    ) -> None:
        self._redis = redis_client
        self._max_attempts = max_attempts
        self._lockout_seconds = lockout_seconds

    async def check(self, client_ip: str) -> tuple[bool, int]:
        key = self._key(client_ip)
        attempts = int(await self._redis.get(key) or 0)
        if attempts < self._max_attempts:
            return True, 0
        return False, max(1, int(await self._redis.ttl(key)))

    async def record_failure(self, client_ip: str) -> None:
        await self._redis.eval(
            _INCREMENT_WITH_EXPIRY,
            1,
            self._key(client_ip),
            self._lockout_seconds,
        )

    async def clear(self, client_ip: str) -> None:
        await self._redis.delete(self._key(client_ip))

    @staticmethod
    def _key(client_ip: str) -> str:
        return f"login-attempts:{client_ip}"
