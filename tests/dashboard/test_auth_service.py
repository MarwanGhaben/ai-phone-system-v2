import hashlib

import pytest

from services.dashboard import auth_service
from services.dashboard.auth_service import AuthService


class FakePool:
    def __init__(self, user_row: dict) -> None:
        self.user_row = user_row
        self.executed: list[tuple[str, tuple]] = []

    async def fetchrow(self, query: str, *args):
        return self.user_row

    async def execute(self, query: str, *args):
        self.executed.append((query, args))


class AllowAllLoginLimiter:
    async def check(self, client_ip: str) -> tuple[bool, int]:
        return True, 0

    async def record_failure(self, client_ip: str) -> None:
        return None

    async def clear(self, client_ip: str) -> None:
        return None


def legacy_hash(password: str, salt: str = "legacy-salt") -> str:
    digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"sha256:{salt}:{digest}"


def test_new_password_hash_uses_bcrypt() -> None:
    password_hash = AuthService().hash_password("correct horse battery staple")

    assert password_hash.startswith(("$2a$", "$2b$", "$2y$"))


def test_malformed_password_hash_is_rejected() -> None:
    assert AuthService().verify_password("password", "not-a-password-hash") is False


@pytest.mark.asyncio
async def test_successful_legacy_login_upgrades_hash(monkeypatch) -> None:
    user_row = {
        "id": 7,
        "username": "admin",
        "email": "admin@example.test",
        "password_hash": legacy_hash("valid-password"),
        "is_active": True,
        "is_superuser": True,
    }
    pool = FakePool(user_row)

    async def get_pool():
        return pool

    monkeypatch.setattr(auth_service, "get_db_pool", get_pool)
    service = AuthService(AllowAllLoginLimiter())

    success, user, error = await service.authenticate_user(
        "admin", "valid-password", "203.0.113.10"
    )

    assert success is True
    assert user["id"] == 7
    assert error == ""
    assert len(pool.executed) == 1
    update_query, update_args = pool.executed[0]
    assert "UPDATE admin_users SET password_hash" in update_query
    assert update_args[0].startswith(("$2a$", "$2b$", "$2y$"))
    assert update_args[1] == 7
