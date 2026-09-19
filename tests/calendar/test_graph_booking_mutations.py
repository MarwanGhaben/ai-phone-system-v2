"""Independent full readback checks for the create-only Graph boundary."""
import asyncio
from contextlib import asynccontextmanager
import json
from datetime import datetime, timezone

from services.calendar.booking_mutations import (BookingMutationRequest, GraphBookingMutations,
                                                 decode_verified_appointment)


def request():
    return BookingMutationRequest(
        "tenant", "business", "service", "staff",
        datetime(2030, 1, 8, 15, tzinfo=timezone.utc),
        datetime(2030, 1, 8, 15, 30, tzinfo=timezone.utc),
        "Synthetic Customer", "+14165550100", "s@example.invalid",
        "Synthetic office", 0, 0,
    )


def wire():
    return {
        "id": "opaque-provider-id", "serviceId": "service",
        "staffMemberIds": ["staff"],
        "start": {"dateTime": "2030-01-08T15:00:00+00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2030-01-08T15:30:00+00:00", "timeZone": "UTC"},
        "duration": "PT30M", "preBuffer": "PT0S", "postBuffer": "PT0S",
        "isLocationOnline": False,
        "serviceLocation": {"displayName": "Synthetic office"},
        "maximumAttendeesCount": 1, "filledAttendeesCount": 1,
        "customerName": "Synthetic Customer", "customerEmailAddress": "s@example.invalid",
        "customerPhone": "+14165550100",
        "customers": [{"name": "Synthetic Customer", "phone": "+14165550100",
                       "emailAddress": "s@example.invalid"}],
    }


def test_complete_exact_readback():
    assert decode_verified_appointment(wire(), "opaque-provider-id", request()) is not None


def test_each_field_and_alias_conflict_rejected():
    baseline = wire()
    for key, value in (("serviceId", "other"), ("staffMemberIds", ["other"]),
                       ("duration", "PT31M"), ("preBuffer", "PT1M"),
                       ("postBuffer", "PT1M"), ("isLocationOnline", True),
                       ("maximumAttendeesCount", 2), ("filledAttendeesCount", 0),
                       ("customerName", "Other"), ("customerPhone", "other"),
                       ("customerEmailAddress", "other@example.invalid"),
                       ("serviceLocation", {"displayName": "Elsewhere"}),
                       ("customers", [{"name": "Other", "phone": "+14165550100",
                                       "emailAddress": "s@example.invalid"}])):
        changed = dict(baseline, **{key: value})
        assert decode_verified_appointment(changed, "opaque-provider-id", request()) is None
    for key in ("duration", "serviceLocation", "customers", "customerEmailAddress"):
        changed = dict(baseline)
        del changed[key]
        assert decode_verified_appointment(changed, "opaque-provider-id", request()) is None
    changed = dict(baseline, startDateTime={"dateTime": "2030-01-08T16:00:00Z", "timeZone": "UTC"},
                   endDateTime=baseline["end"])
    assert decode_verified_appointment(changed, "opaque-provider-id", request()) is None
    for key in ("start", "end"):
        changed = dict(baseline)
        changed[key] = {"dateTime": "2030-01-08T15:00:00+00:60", "timeZone": "UTC"}
        assert decode_verified_appointment(changed, "opaque-provider-id", request()) is None
    changed = dict(baseline, startDateTime=baseline["start"])
    assert decode_verified_appointment(changed, "opaque-provider-id", request()) is None


class SyntheticResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self.payload = json.dumps(payload).encode()

    async def aiter_bytes(self):
        yield self.payload


class SyntheticHttp:
    def __init__(self, *, token_status=200, create_status=201, create_wire=None,
                 read_status=200, read_wire=None):
        self.calls = []
        self.token_status = token_status
        self.create_status = create_status
        self.create_wire = {"id": "opaque-provider-id"} if create_wire is None else create_wire
        self.read_status = read_status
        self.read_wire = read_wire or wire()

    @asynccontextmanager
    async def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if "login.microsoftonline.com" in url:
            response = SyntheticResponse(self.token_status, {"access_token": "synthetic-token"})
        elif method == "POST":
            response = SyntheticResponse(self.create_status, self.create_wire)
        else:
            response = SyntheticResponse(self.read_status, self.read_wire)
        yield response


def test_actual_transport_separates_post_and_get():
    async def run():
        http = SyntheticHttp()
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                                        client_id="client", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        payload = json.loads(prepared.body)
        assert payload["start"]["dateTime"] == "2030-01-08T15:00:00Z"
        assert payload["end"]["dateTime"] == "2030-01-08T15:30:00Z"
        assert payload["smsNotificationsEnabled"] is False
        receipt = await adapter.create_once(prepared)
        assert receipt.status == "receipt" and receipt.provider_id == "opaque-provider-id"
        verified = await adapter.read_exact(receipt.provider_id, request())
        assert verified is not None
        graph = [(method, url, args) for method, url, args in http.calls if "graph.microsoft.com" in url]
        assert [method for method, _, _ in graph] == ["POST", "GET"]
        assert json.loads(graph[0][2]["content"])["staffMemberIds"] == ["staff"]
        assert graph[1][1].endswith("/appointments/opaque-provider-id")
    asyncio.run(run())


def test_http_failure_has_no_create_retry():
    async def run():
        http = SyntheticHttp(create_status=429)
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                                        client_id="client", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        result = await adapter.create_once(prepared)
        assert result.status == "uncertain"
        assert sum(method == "POST" and "graph.microsoft.com" in url
                   for method, url, _ in http.calls) == 1
    asyncio.run(run())


def test_token_failure_has_zero_graph_posts():
    async def run():
        http = SyntheticHttp(token_status=401)
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                                        client_id="client", client_secret="synthetic", client=http)
        assert await adapter.prepare(request()) is None
        assert not any("graph.microsoft.com" in url for _, url, _ in http.calls)
    asyncio.run(run())


def test_oversized_201_is_uncertain_without_retry():
    async def run():
        http = SyntheticHttp(create_wire={"id": "opaque-provider-id", "extra": "x" * 300000})
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                                        client_id="client", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        assert (await adapter.create_once(prepared)).status == "uncertain"
        assert sum(method == "POST" and "graph.microsoft.com" in url
                   for method, url, _ in http.calls) == 1
    asyncio.run(run())


def test_documented_seven_digit_zero_fraction_and_exactness():
    complete = wire()
    complete["start"] = {"dateTime": "2030-01-08T15:00:00.0000000Z", "timeZone": "UTC"}
    complete["end"] = {"dateTime": "2030-01-08T15:30:00.0000000Z", "timeZone": "UTC"}
    assert decode_verified_appointment(complete, "opaque-provider-id", request()) is not None
    complete["end"] = {"dateTime": "2030-01-08T15:30:00.0000001Z", "timeZone": "UTC"}
    assert decode_verified_appointment(complete, "opaque-provider-id", request()) is None
    complete["end"] = {"dateTime": "2030-01-08T15:30:00+00:60", "timeZone": "UTC"}
    assert decode_verified_appointment(complete, "opaque-provider-id", request()) is None
    complete["end"] = wire()["end"]
    complete["startDateTime"] = {"dateTime": "2030-01-08T15:00:00.0000000Z", "timeZone": "UTC"}
    complete["endDateTime"] = {"dateTime": "2030-01-08T15:31:00Z", "timeZone": "UTC"}
    assert decode_verified_appointment(complete, "opaque-provider-id", request()) is None


def test_cancel_resistant_stream_exit_has_bounded_public_return():
    async def run():
        entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class SlowClose(SyntheticHttp):
            @asynccontextmanager
            async def stream(self, method, url, **kwargs):
                try:
                    yield SyntheticResponse(200, {"access_token": "synthetic"})
                finally:
                    entered.set()
                    while not release.is_set():
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            continue
                    closed.set()

        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=SlowClose())
        from unittest import mock
        original_timeout = asyncio.timeout
        with mock.patch("asyncio.timeout", side_effect=lambda seconds: original_timeout(min(seconds, .02))):
            task = asyncio.create_task(adapter.prepare(request()))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(.15)
                assert task.done()
            finally:
                release.set()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
                if adapter._retained:
                    await asyncio.wait_for(asyncio.gather(*tuple(adapter._retained),
                                                          return_exceptions=True), 2)
                assert closed.is_set()
    asyncio.run(run())


def test_saturated_admission_cannot_start_late_create():
    async def run():
        release = asyncio.Event()

        class ResistantGate:
            async def __aenter__(self):
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue

            async def __aexit__(self, *args):
                pass

        http = SyntheticHttp()
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        adapter._slots = ResistantGate()
        from unittest import mock
        original_timeout = asyncio.timeout
        with mock.patch("asyncio.timeout", side_effect=lambda seconds: original_timeout(min(seconds, .02))):
            tasks = [asyncio.create_task(adapter.create_once(prepared)) for _ in range(9)]
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
                assert all(item.status != "receipt" for item in results)
            finally:
                release.set()
                if adapter._retained:
                    await asyncio.wait_for(asyncio.gather(*tuple(adapter._retained),
                                                          return_exceptions=True), 2)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(method == "POST" and "graph.microsoft.com" in url
                   for method, url, _ in http.calls) == 0
        assert not adapter._retained
    asyncio.run(run())


def test_cancelled_admission_releases_without_late_create():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class ResistantGate:
            async def __aenter__(self):
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue

            async def __aexit__(self, *args):
                pass

        http = SyntheticHttp()
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        adapter._slots = ResistantGate()
        task = asyncio.create_task(adapter.create_once(prepared))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
        finally:
            release.set()
            if adapter._retained:
                await asyncio.wait_for(asyncio.gather(*tuple(adapter._retained),
                                                      return_exceptions=True), 2)
            assert sum(method == "POST" and "graph.microsoft.com" in url
                       for method, url, _ in http.calls) == 0
            assert not adapter._retained
    asyncio.run(run())


def test_owned_client_close_is_bounded_and_injected_client_is_preserved():
    async def run():
        entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class SlowClient(SyntheticHttp):
            close_calls = 0

            async def aclose(self):
                self.close_calls += 1
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                closed.set()

        owned = SlowClient()
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=owned)
        adapter._owns = True
        from unittest import mock
        original_timeout = asyncio.timeout
        with mock.patch("asyncio.timeout", side_effect=lambda seconds: original_timeout(min(seconds, .02))):
            task = asyncio.create_task(adapter.close())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(.15)
                assert task.done()
            finally:
                release.set()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
                if adapter._retained:
                    await asyncio.wait_for(asyncio.gather(*tuple(adapter._retained),
                                                          return_exceptions=True), 2)
                assert closed.is_set()
        injected = SlowClient()
        other = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=injected)
        await other.close()
        assert injected.close_calls == 0
    asyncio.run(run())
