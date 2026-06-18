from collections import defaultdict

import pytest
from starlette.requests import Request

from services.dashboard import auth_service
from services.dashboard.auth_service import AuthService
from services.security.client_ip import ClientIPResolver
from services.security.rate_limiter import LoginAttemptLimiter, RedisRateLimiter


class FakeRedis:
    def __init__(self) -> None:
        self.counts = defaultdict(int)
        self.expirations: dict[str, int] = {}

    async def eval(self, script: str, key_count: int, key: str, ttl: int):
        self.counts[key] += 1
        self.expirations.setdefault(key, int(ttl))
        return [self.counts[key], self.expirations[key]]

    async def get(self, key: str):
        count = self.counts.get(key)
        return str(count) if count is not None else None

    async def ttl(self, key: str) -> int:
        return self.expirations.get(key, -2)

    async def delete(self, key: str) -> int:
        existed = key in self.counts
        self.counts.pop(key, None)
        self.expirations.pop(key, None)
        return int(existed)


def make_request(peer: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "client": (peer, 12345),
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "http_version": "1.1",
        }
    )


def test_untrusted_peer_cannot_spoof_forwarded_address() -> None:
    resolver = ClientIPResolver(["172.16.0.0/12"])

    assert resolver.resolve(make_request("203.0.113.4", "198.51.100.9")) == "203.0.113.4"


def test_trusted_proxy_supplies_client_address() -> None:
    resolver = ClientIPResolver(["172.16.0.0/12"])

    assert resolver.resolve(make_request("172.18.0.3", "198.51.100.9")) == "198.51.100.9"


@pytest.mark.asyncio
async def test_rate_limit_counters_are_atomic_and_expire() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis, requests_per_minute=3, burst_limit=2)

    first_limited, _ = await limiter.check("198.51.100.9", now=120.1)
    second_limited, _ = await limiter.check("198.51.100.9", now=120.2)
    third_limited, details = await limiter.check("198.51.100.9", now=120.3)

    assert first_limited is False
    assert second_limited is False
    assert third_limited is True
    assert details["reason"] == "burst_limit_exceeded"
    assert all(ttl > 0 for ttl in redis.expirations.values())


class MissingUserPool:
    async def fetchrow(self, query: str, *args):
        return None


@pytest.mark.asyncio
async def test_login_attempts_are_shared_between_auth_instances(monkeypatch) -> None:
    redis = FakeRedis()
    first_auth = AuthService(LoginAttemptLimiter(redis, max_attempts=2, lockout_seconds=900))
    second_auth = AuthService(LoginAttemptLimiter(redis, max_attempts=2, lockout_seconds=900))

    async def get_pool():
        return MissingUserPool()

    monkeypatch.setattr(auth_service, "get_db_pool", get_pool)

    await first_auth.authenticate_user("missing", "bad", "198.51.100.9")
    await second_auth.authenticate_user("missing", "bad", "198.51.100.9")
    success, _, message = await first_auth.authenticate_user(
        "missing", "bad", "198.51.100.9"
    )

    assert success is False
    assert message.startswith("Too many login attempts")
