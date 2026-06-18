import json

import pytest
from starlette.requests import Request

from services.dashboard import dashboard_routes, email_service
from services.dashboard.dashboard_routes import LoginRequest, MFARequest
from services.dashboard.email_service import EmailService


def dashboard_request(cookie: str = "") -> Request:
    headers = [(b"cookie", cookie.encode())] if cookie else []
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/dashboard/api/login",
            "headers": headers,
            "client": ("203.0.113.10", 12345),
            "query_string": b"",
            "server": ("testserver", 443),
            "scheme": "https",
            "http_version": "1.1",
        }
    )


def clear_graph_environment(monkeypatch) -> None:
    for variable in (
        "MSGRAPH_TENANT_ID",
        "MSGRAPH_CLIENT_ID",
        "MSGRAPH_CLIENT_SECRET",
        "MSGRAPH_SENDER_EMAIL",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.asyncio
async def test_unconfigured_mfa_email_fails_without_logging_code(monkeypatch) -> None:
    clear_graph_environment(monkeypatch)
    warning_messages: list[str] = []
    monkeypatch.setattr(
        email_service.logger,
        "warning",
        lambda message: warning_messages.append(str(message)),
    )
    service = EmailService()

    delivered = await service.send_mfa_code(
        "admin@example.test", "123456", "Admin"
    )

    assert delivered is False
    assert all("123456" not in message for message in warning_messages)


def test_email_service_has_no_optimistic_sync_wrapper() -> None:
    assert not hasattr(EmailService, "send_email")


@pytest.mark.asyncio
async def test_login_alert_escapes_request_metadata(monkeypatch) -> None:
    monkeypatch.setenv("MSGRAPH_TENANT_ID", "tenant")
    monkeypatch.setenv("MSGRAPH_CLIENT_ID", "client")
    monkeypatch.setenv("MSGRAPH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("MSGRAPH_SENDER_EMAIL", "sender@example.test")
    service = EmailService()
    captured_html = ""

    async def capture_email(to_email: str, subject: str, body_html: str) -> bool:
        nonlocal captured_html
        captured_html = body_html
        return True

    service.send_email_async = capture_email

    delivered = await service.send_login_alert(
        "admin@example.test",
        "Admin",
        "203.0.113.10",
        '<img src=x onerror="alert(1)">',
    )

    assert delivered is True
    assert "<img" not in captured_html
    assert "&lt;img" in captured_html


class FailedMFAAuth:
    def __init__(self) -> None:
        self.invalidated_users: list[int] = []

    async def authenticate_user(self, username, password, ip_address):
        return True, {
            "id": 7,
            "username": "admin",
            "email": "admin@example.test",
        }, ""

    async def create_mfa_code(self, user_id: int) -> str:
        return "123456"

    async def invalidate_mfa_codes(self, user_id: int) -> None:
        self.invalidated_users.append(user_id)


class FailedMFAEmail:
    async def send_mfa_code(self, to_email: str, code: str, username: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_login_fails_closed_when_mfa_delivery_fails(monkeypatch) -> None:
    auth = FailedMFAAuth()
    monkeypatch.setattr(dashboard_routes, "get_auth_service", lambda: auth)
    monkeypatch.setattr(dashboard_routes, "get_email_service", FailedMFAEmail)
    monkeypatch.setattr(
        dashboard_routes, "get_client_ip", lambda request: "203.0.113.10"
    )

    response = await dashboard_routes.login(
        dashboard_request(), LoginRequest(username="admin", password="password")
    )

    assert response.status_code == 503
    assert auth.invalidated_users == [7]
    assert b"mfa_pending" not in response.headers.get("set-cookie", "").encode()


class VerifiedMFAAuth:
    async def verify_mfa_code(self, user_id: int, code: str):
        return True, ""

    async def get_user_by_id(self, user_id: int):
        return {
            "id": user_id,
            "username": "admin",
            "email": "admin@example.test",
        }

    async def create_session(self, user_id: int, ip_address: str, user_agent: str):
        return "session-token"


class FailedAlertEmail:
    def __init__(self) -> None:
        self.attempts = 0

    async def send_login_alert(self, *args) -> bool:
        self.attempts += 1
        return False


@pytest.mark.asyncio
async def test_failed_login_alert_does_not_revoke_verified_session(monkeypatch) -> None:
    alert_email = FailedAlertEmail()
    monkeypatch.setattr(
        dashboard_routes, "get_auth_service", lambda: VerifiedMFAAuth()
    )
    monkeypatch.setattr(dashboard_routes, "get_email_service", lambda: alert_email)
    monkeypatch.setattr(
        dashboard_routes, "get_client_ip", lambda request: "203.0.113.10"
    )

    response = await dashboard_routes.verify_mfa(
        dashboard_request("mfa_pending=7"), MFARequest(code="123456")
    )

    assert response.status_code == 200
    assert json.loads(response.body)["success"] is True
    assert alert_email.attempts == 1
