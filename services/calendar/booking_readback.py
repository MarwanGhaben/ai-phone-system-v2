"""Bounded, read-only Microsoft Bookings appointment observation."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
import time
from urllib.parse import quote


TOKEN_HOST = "https://login.microsoftonline.com"
GRAPH_HOST = "https://graph.microsoft.com"
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?(?:Z|[+-]\d{2}:\d{2})?$")
_CATEGORIES = frozenset({
    "configuration", "authentication", "authorization", "throttled",
    "server_error", "timeout", "network", "malformed", "identity_mismatch",
    "http_error",
})


class ReadbackFailure(Exception):
    """Only fixed categories, never provider content, cross this boundary."""

    def __init__(self, category: str, *, http_status: int | None = None,
                 retry_after: int = 0, stop_batch: bool = False):
        if category not in _CATEGORIES:
            category = "malformed"
        super().__init__(category)
        self.category = category
        self.http_status = http_status
        self.retry_after = retry_after
        self.stop_batch = stop_batch


@dataclass(frozen=True)
class AppointmentDetails:
    start: datetime
    end: datetime
    staff_member_ids: tuple[str, ...] | None
    service_id: str | None
    is_location_online: bool | None


@dataclass(frozen=True)
class ProviderResult:
    outcome: str
    error_category: str | None = None
    http_status: int | None = None
    details: AppointmentDetails | None = None
    retry_after: int = 0
    stop_batch: bool = False
    observed_provider_id: str | None = None


def _utc(value: object) -> datetime:
    if not isinstance(value, dict) or value.get("timeZone") != "UTC":
        raise ReadbackFailure("malformed")
    text = value.get("dateTime")
    if not isinstance(text, str) or not _DATETIME.fullmatch(text):
        raise ReadbackFailure("malformed")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        offset = parsed.utcoffset()
    except (TypeError, ValueError, OverflowError):
        raise ReadbackFailure("malformed") from None
    if offset is None:
        return parsed.replace(tzinfo=timezone.utc)
    if offset != timezone.utc.utcoffset(None):
        raise ReadbackFailure("malformed")
    return parsed.astimezone(timezone.utc)


def parse_appointment(payload: object, saved_provider_id: str) -> AppointmentDetails:
    """Parse complete official/observed UTC pairs with exact identity."""
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise ReadbackFailure("malformed")
    if payload["id"] != saved_provider_id:
        raise ReadbackFailure("identity_mismatch")
    pairs = []
    for first, last in (("start", "end"), ("startDateTime", "endDateTime")):
        if (first in payload) != (last in payload):
            raise ReadbackFailure("malformed")
        if first in payload:
            start, end = _utc(payload[first]), _utc(payload[last])
            if end <= start:
                raise ReadbackFailure("malformed")
            pairs.append((start, end))
    if not pairs or (len(pairs) == 2 and pairs[0] != pairs[1]):
        raise ReadbackFailure("malformed")
    staff = payload.get("staffMemberIds")
    if staff is not None:
        if (not isinstance(staff, list)
                or not all(isinstance(item, str) and item for item in staff)):
            raise ReadbackFailure("malformed")
        staff = tuple(staff)
    service = payload.get("serviceId")
    if service is not None and not isinstance(service, str):
        raise ReadbackFailure("malformed")
    online = payload.get("isLocationOnline")
    if online is not None and not isinstance(online, bool):
        raise ReadbackFailure("malformed")
    return AppointmentDetails(pairs[0][0], pairs[0][1], staff, service, online)


def _retry_after(headers: object) -> int:
    raw = None
    try:
        raw = headers.get("Retry-After")
        if raw is None:
            return 60
        seconds = int(raw)
        if seconds < 0:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        try:
            seconds = math.ceil((parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            seconds = 60
    return max(1, seconds)


def _http_failure(status: int, headers: object, *, token: bool) -> ReadbackFailure:
    if status == 429:
        return ReadbackFailure("throttled", http_status=status,
                               retry_after=_retry_after(headers), stop_batch=True)
    if status in (401, 403):
        return ReadbackFailure("authentication" if status == 401 else "authorization",
                               http_status=status, retry_after=_retry_after(headers),
                               stop_batch=True)
    if status >= 500:
        return ReadbackFailure("server_error", http_status=status,
                               retry_after=_retry_after(headers), stop_batch=True)
    return ReadbackFailure("http_error", http_status=status,
                           retry_after=60 if token else 0, stop_batch=token)


class BookingReadbackClient:
    """A token-caching exact-ID GET client; it never uses the mutation adapter."""

    def __init__(self, settings: object, *, client: object | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._token: str | None = None
        self._token_until = 0.0

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
        self._token = None

    async def _transport(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=False)
        return self._client

    async def _access_token(self) -> str:
        if self._token is not None and time.monotonic() < self._token_until:
            return self._token
        s = self.settings
        if not all((s.ms_bookings_tenant_id, s.ms_bookings_client_id,
                    s.ms_bookings_client_secret, s.ms_bookings_business_id)):
            raise ReadbackFailure("configuration", stop_batch=True)
        client = await self._transport()
        try:
            response = await client.post(
                TOKEN_HOST + "/" + quote(s.ms_bookings_tenant_id, safe="") + "/oauth2/v2.0/token",
                data={"grant_type": "client_credentials", "client_id": s.ms_bookings_client_id,
                      "client_secret": s.ms_bookings_client_secret,
                      "scope": "https://graph.microsoft.com/.default"},
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            try:
                import httpx
                timeout = isinstance(error, (httpx.TimeoutException, TimeoutError))
            except ImportError:
                timeout = isinstance(error, TimeoutError)
            raise ReadbackFailure(
                "timeout" if timeout else "network",
                retry_after=60, stop_batch=True) from None
        if response.status_code != 200:
            raise _http_failure(response.status_code, response.headers, token=True)
        try:
            body = response.json()
            token = body["access_token"]
            lifetime = body.get("expires_in", 300)
        except (ValueError, TypeError, KeyError):
            raise ReadbackFailure("malformed", stop_batch=True) from None
        try:
            finite_lifetime = math.isfinite(lifetime) and lifetime > 0
        except (TypeError, ValueError, OverflowError):
            finite_lifetime = False
        if (not isinstance(token, str) or not token or not isinstance(lifetime, (int, float))
                or isinstance(lifetime, bool) or not finite_lifetime):
            raise ReadbackFailure("malformed", stop_batch=True)
        self._token = token
        self._token_until = time.monotonic() + max(1, min(float(lifetime), 3600) - 60)
        return token

    async def observe(self, saved_provider_id: str, local_start: datetime) -> ProviderResult:
        """Return a fixed outcome; cancellation remains visible to the poller."""
        try:
            if (not isinstance(saved_provider_id, str) or not saved_provider_id
                    or not isinstance(local_start, datetime) or local_start.utcoffset() is None):
                raise ReadbackFailure("configuration", stop_batch=True)
            token = await self._access_token()
            client = await self._transport()
            url = (GRAPH_HOST + "/v1.0/solutions/bookingBusinesses/"
                   + quote(self.settings.ms_bookings_business_id, safe="")
                   + "/appointments/" + quote(saved_provider_id, safe=""))
            response = await client.get(url, headers={"Authorization": "Bearer " + token})
            if response.status_code == 404:
                return ProviderResult("unavailable", http_status=404)
            if response.status_code != 200:
                if response.status_code == 401:
                    self._token = None
                    self._token_until = 0.0
                raise _http_failure(response.status_code, response.headers, token=False)
            try:
                body = response.json()
            except (ValueError, TypeError):
                raise ReadbackFailure("malformed") from None
            details = parse_appointment(body, saved_provider_id)
            outcome = "present" if details.start == local_start.astimezone(timezone.utc) else "changed"
            return ProviderResult(
                outcome, http_status=200, details=details,
                observed_provider_id=body["id"])
        except asyncio.CancelledError:
            raise
        except ReadbackFailure as error:
            return ProviderResult("check_failed", error.category, error.http_status,
                                  retry_after=error.retry_after, stop_batch=error.stop_batch)
        except Exception as error:
            # Never log the exception: HTTP exceptions may contain the opaque URL.
            try:
                import httpx
                timeout = isinstance(error, (httpx.TimeoutException, TimeoutError))
            except ImportError:
                timeout = isinstance(error, TimeoutError)
            return ProviderResult("check_failed", "timeout" if timeout else "network")
