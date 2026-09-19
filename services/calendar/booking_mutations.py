"""Single-attempt Bookings create and strict exact-ID verification.

This boundary has no retry path. It never treats a POST body as a certificate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Awaitable, Callable
from urllib.parse import quote

from services.calendar.service_facts import _duration


_TIME = re.compile(r"(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,7}))?(?P<offset>Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z")
_HOST = "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/"
_TOKEN_HOST = "https://login.microsoftonline.com/"


def _valid(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and value == value.strip() and len(value) <= 255


def _provider_id(value: object) -> bool:
    return _valid(value) and all(ord(character) >= 32 and ord(character) != 127 for character in value)


def _instant(value: object) -> datetime | None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _time(value: object) -> datetime:
    if (not isinstance(value, dict) or value.get("timeZone") != "UTC"
            or not isinstance(value.get("dateTime"), str)):
        raise ValueError
    match = _TIME.fullmatch(value["dateTime"])
    if match is None:
        raise ValueError
    fraction = match.group("fraction")
    if fraction is not None and len(fraction) == 7:
        if fraction[-1] != "0":
            raise ValueError
        fraction = fraction[:6]
    normalized = (match.group("base") + ("." + fraction if fraction else "")
                  + ("+00:00" if match.group("offset") == "Z" else match.group("offset")))
    instant = datetime.fromisoformat(normalized)
    if instant.utcoffset() != timedelta(0):
        raise ValueError
    return instant.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True, repr=False)
class BookingMutationRequest:
    tenant_id: str
    business_id: str
    service_id: str
    staff_id: str
    start: datetime
    end: datetime
    customer_name: str
    customer_phone: str
    customer_email: str
    location: str
    pre_buffer_seconds: int
    post_buffer_seconds: int

    def __post_init__(self) -> None:
        if (not all(_valid(item) for item in (self.tenant_id, self.business_id,
                self.service_id, self.staff_id, self.customer_name, self.customer_phone,
                self.location)) or type(self.customer_email) is not str
                or len(self.customer_email) > 255 or self.customer_email != self.customer_email.strip()
                or _instant(self.start) is None or _instant(self.end) is None
                or self.end - self.start != timedelta(minutes=30)
                or type(self.pre_buffer_seconds) is not int or type(self.post_buffer_seconds) is not int
                or not 0 <= self.pre_buffer_seconds <= 7200
                or not 0 <= self.post_buffer_seconds <= 7200):
            raise ValueError("invalid booking mutation request")

    def payload(self) -> dict:
        def graph_time(value: datetime) -> dict:
            return {"dateTime": value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "timeZone": "UTC"}
        return {
            "@odata.type": "#microsoft.graph.bookingAppointment",
            "serviceId": self.service_id, "staffMemberIds": [self.staff_id],
            "start": graph_time(self.start), "end": graph_time(self.end),
            "duration": "PT30M", "preBuffer": f"PT{self.pre_buffer_seconds}S",
            "postBuffer": f"PT{self.post_buffer_seconds}S",
            "isLocationOnline": False,
            "serviceLocation": {"displayName": self.location},
            "maximumAttendeesCount": 1,
            "customerName": self.customer_name,
            "customerPhone": self.customer_phone,
            "customerEmailAddress": self.customer_email,
            "customerTimeZone": "America/Toronto",
            "customers": [{"name": self.customer_name, "phone": self.customer_phone,
                           "emailAddress": self.customer_email, "timeZone": "America/Toronto"}],
            "optOutOfCustomerEmail": False, "smsNotificationsEnabled": False,
        }


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedAppointment:
    provider_id: str
    request: BookingMutationRequest


def decode_verified_appointment(wire: object, provider_id: str,
                                request: BookingMutationRequest) -> VerifiedAppointment | None:
    """Require every agreed field. Only exact time aliases and harmless UTC spellings normalize."""
    try:
        if type(wire) is not dict or not _provider_id(provider_id) or wire.get("id") != provider_id:
            raise ValueError
        pairs = []
        for start_key, end_key in (("start", "end"), ("startDateTime", "endDateTime")):
            if (start_key in wire) != (end_key in wire):
                raise ValueError
            if start_key in wire:
                pairs.append((_time(wire[start_key]), _time(wire[end_key])))
        if not pairs or any(pair != pairs[0] for pair in pairs):
            raise ValueError
        if pairs[0] != (_instant(request.start), _instant(request.end)):
            raise ValueError
        if (wire["serviceId"] != request.service_id
                or wire["staffMemberIds"] != [request.staff_id]
                or _duration(wire["duration"]) != request.end - request.start
                or _duration(wire["preBuffer"]) != timedelta(seconds=request.pre_buffer_seconds)
                or _duration(wire["postBuffer"]) != timedelta(seconds=request.post_buffer_seconds)
                or wire["isLocationOnline"] is not False
                or wire["serviceLocation"]["displayName"] != request.location
                or type(wire["maximumAttendeesCount"]) is not int
                or wire["maximumAttendeesCount"] != 1
                or type(wire["filledAttendeesCount"]) is not int
                or wire["filledAttendeesCount"] != 1
                or wire["customerName"] != request.customer_name
                or wire["customerPhone"] != request.customer_phone
                or wire["customerEmailAddress"] != request.customer_email):
            raise ValueError
        customers = wire["customers"]
        if (type(customers) is not list or len(customers) != 1 or type(customers[0]) is not dict
                or customers[0].get("name") != request.customer_name
                or customers[0].get("phone") != request.customer_phone
                or customers[0].get("emailAddress") != request.customer_email):
            raise ValueError
        return VerifiedAppointment(provider_id, request)
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError):
        return None


@dataclass(frozen=True, slots=True, repr=False)
class MutationResponse:
    status: str
    provider_id: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class PreparedMutation:
    request: BookingMutationRequest
    token: str
    body: bytes


class _Permit:
    __slots__ = ("active", "deadline")

    def __init__(self, deadline: float) -> None:
        self.active = True
        self.deadline = deadline

    def current(self) -> bool:
        return self.active and asyncio.get_running_loop().time() < self.deadline


class GraphBookingMutations:
    """Scoped HTTP transport. Inject a synthetic httpx client for tests."""

    def __init__(self, *, tenant_id: str, business_id: str, client_id: str,
                 client_secret: str, client=None):
        if not all(_valid(item) for item in (tenant_id, business_id, client_id, client_secret)):
            raise ValueError("invalid Graph configuration")
        self.tenant_id, self.business_id = tenant_id, business_id
        self._client_id, self._secret = client_id, client_secret
        self._client = client
        self._owns = client is None
        self._slots = asyncio.Semaphore(8)
        self._admission = asyncio.Semaphore(8)
        self._retained: set[asyncio.Task] = set()
        self._closed = False

    def _url(self, provider_id: str | None = None) -> str:
        url = _HOST + quote(self.business_id, safe="") + "/appointments"
        return url if provider_id is None else url + "/" + quote(provider_id, safe="")

    async def _http(self):
        if self._closed:
            raise RuntimeError("transport closed")
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(follow_redirects=False, timeout=httpx.Timeout(10))
        return self._client

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is None or not self._owns:
            return
        task = asyncio.create_task(client.aclose())
        self._track(task)
        try:
            async with asyncio.timeout(2):
                await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception:
            task.cancel()

    def _track(self, task: asyncio.Task) -> None:
        self._retained.add(task)

        def finished(done: asyncio.Task) -> None:
            self._retained.discard(done)
            try:
                done.result()
            except BaseException:
                pass

        task.add_done_callback(finished)

    async def _owned(self, work):
        """Bound public wait; retain an unfinished worker and its admission slot."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 20
        acquired = False
        try:
            async with asyncio.timeout(20):
                await self._admission.acquire()
                acquired = True
        except BaseException:
            if acquired:
                self._admission.release()
            raise
        permit = _Permit(deadline)

        async def run():
            try:
                return await work(permit)
            finally:
                self._admission.release()

        try:
            task = asyncio.create_task(run())
        except BaseException:
            self._admission.release()
            raise
        self._track(task)
        try:
            async with asyncio.timeout(max(0, deadline - loop.time())):
                return await asyncio.shield(task)
        except BaseException:
            permit.active = False
            # Keep the worker and admission slot until its response/stream exit
            # completes. A late admitted worker sees the revoked permit before
            # it may start network I/O, and a sent create remains uncertain.
            task.cancel()
            raise

    async def _request(self, method: str, url: str, *, limit: int, **kwargs):
        client = await self._http()
        async with client.stream(method, url, follow_redirects=False, **kwargs) as response:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > limit:
                    raise ValueError("oversized response")
            return response.status_code, bytes(body)

    async def prepare(self, request: BookingMutationRequest) -> PreparedMutation | None:
        if (type(request) is not BookingMutationRequest or request.tenant_id != self.tenant_id
                or request.business_id != self.business_id):
            return None
        payload = request.payload()
        try:
            payload_body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")).encode("utf-8")
            if len(payload_body) > 65536:
                return None
            async def token_request(permit: _Permit):
                async with self._slots:
                    if not permit.current():
                        return None
                    return await self._request("POST", _TOKEN_HOST + quote(self.tenant_id, safe="")
                        + "/oauth2/v2.0/token", limit=65536,
                        data={"grant_type": "client_credentials", "client_id": self._client_id,
                              "client_secret": self._secret,
                              "scope": "https://graph.microsoft.com/.default"})
            result = await self._owned(token_request)
            if result is None:
                return None
            status, body = result
            if status != 200:
                return None
            token = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))["access_token"]
            if type(token) is not str or not token or len(token) > 16384:
                return None
            return PreparedMutation(request, token, payload_body)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def create_once(self, prepared: PreparedMutation, *,
                          before_send: Callable[[], Awaitable[bool]] | None = None) -> MutationResponse:
        if (type(prepared) is not PreparedMutation
                or prepared.request.tenant_id != self.tenant_id
                or prepared.request.business_id != self.business_id):
            return MutationResponse("invalid")
        try:
            async def create_request(permit: _Permit):
                async with self._slots:
                    if not permit.current():
                        return None
                    if before_send is not None:
                        try:
                            allowed = await before_send()
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            return None
                        if allowed is not True or not permit.current():
                            return None
                    return await self._request("POST", self._url(), limit=262144,
                        headers={"Authorization": "Bearer " + prepared.token,
                                 "Content-Type": "application/json"}, content=prepared.body)
            result = await self._owned(create_request)
            if result is None:
                return MutationResponse("not_sent")
            status, body = result
            if status != 201:
                return MutationResponse("uncertain")
            wire = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            provider_id = wire.get("id") if type(wire) is dict else None
            return (MutationResponse("receipt", provider_id) if _provider_id(provider_id)
                    else MutationResponse("uncertain"))
        except asyncio.CancelledError:
            raise
        except Exception:
            return MutationResponse("uncertain")

    async def read_exact(self, provider_id: str, request: BookingMutationRequest) -> VerifiedAppointment | None:
        if not _provider_id(provider_id) or request.tenant_id != self.tenant_id or request.business_id != self.business_id:
            return None
        prepared = await self.prepare(request)
        if prepared is None:
            return None
        try:
            async def get_request(permit: _Permit):
                async with self._slots:
                    if not permit.current():
                        return None
                    return await self._request("GET", self._url(provider_id), limit=262144,
                        headers={"Authorization": "Bearer " + prepared.token})
            result = await self._owned(get_request)
            if result is None:
                return None
            status, body = result
            if status != 200:
                return None
            wire = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            return decode_verified_appointment(wire, provider_id, request)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
