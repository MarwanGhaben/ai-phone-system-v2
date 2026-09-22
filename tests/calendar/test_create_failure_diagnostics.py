"""Create failures must remain single-attempt and leave usable, private evidence."""
import asyncio
from contextlib import asynccontextmanager
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import AsyncMock
from uuid import uuid4

from loguru import logger
import pytest

from services.calendar.booking_mutations import GraphBookingMutations
from tests.calendar.test_graph_booking_mutations import SyntheticHttp, request


@pytest.mark.parametrize("status,body,reason,code", [
    (400, {"error": {"code": "BadRequest", "message": "PRIVATE-CUSTOMER"}},
     "http_rejected", "BadRequest"),
    (403, {"error": {"code": "PRIVATE-SECRET", "message": "PRIVATE-CUSTOMER"}},
     "http_rejected", "unclassified"),
    (429, {"error": {"code": "TooManyRequests"}}, "http_rejected", "TooManyRequests"),
    (503, {"error": "PRIVATE-CUSTOMER"}, "http_rejected", "unclassified"),
    (201, {"not_id": "PRIVATE-PROVIDER-ID"}, "receipt_invalid", "none"),
    (201, {"id": "PRIVATE-PROVIDER-ID"}, "receipt_received", "none"),
])
def test_actual_transport_keeps_safe_http_evidence_without_retry(status, body, reason, code):
    async def run():
        messages = []
        sink = logger.add(lambda entry: messages.append(str(entry)), format="{message}")
        try:
            http = SyntheticHttp(create_status=status, create_wire=body)
            adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                client_id="client", client_secret="PRIVATE-SECRET", client=http)
            prepared = await adapter.prepare(request())
            result = await adapter.create_once(prepared)
        finally:
            logger.remove(sink)
        assert result.status == ("receipt" if reason == "receipt_received" else "uncertain")
        assert sum(m == "POST" and "graph.microsoft.com" in u for m, u, _ in http.calls) == 1
        evidence = [json.loads(m.split("BookingCreateDiagnostic ", 1)[1])
                    for m in messages if "BookingCreateDiagnostic " in m]
        assert any(e == {"stage": "transport", "reason": reason,
                         "http_status": status, "provider_code": code} for e in evidence)
        assert "PRIVATE-" not in "".join(messages)
        assert "synthetic-token" not in "".join(messages)
    asyncio.run(run())


def test_denied_send_is_reported_without_contacting_graph():
    async def run():
        messages = []
        sink = logger.add(lambda entry: messages.append(str(entry)), format="{message}")
        try:
            http = SyntheticHttp()
            adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                client_id="client", client_secret="synthetic", client=http)
            prepared = await adapter.prepare(request())
            async def deny():
                return False
            result = await adapter.create_once(prepared, before_send=deny)
        finally:
            logger.remove(sink)
        assert result.status == "not_sent"
        assert not any("graph.microsoft.com" in url for _, url, _ in http.calls)
        assert '"reason": "not_sent"' in "".join(messages)
    asyncio.run(run())


def test_diagnostic_sink_failure_cannot_change_receipt_or_retry():
    async def run():
        http = SyntheticHttp()
        adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="client", client_secret="synthetic", client=http)
        prepared = await adapter.prepare(request())
        with patch.object(logger, "info", side_effect=RuntimeError("PRIVATE-SINK")):
            assert (await adapter.create_once(prepared)).status == "receipt"
        assert sum(m == "POST" and "graph.microsoft.com" in u for m, u, _ in http.calls) == 1
    asyncio.run(run())


def test_cancellation_is_reported_and_still_propagates():
    class Cancelled(SyntheticHttp):
        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            if "graph.microsoft.com" in url:
                raise asyncio.CancelledError
            async with super().stream(method, url, **kwargs) as response:
                yield response

    async def run():
        messages = []
        sink = logger.add(lambda entry: messages.append(str(entry)), format="{message}")
        try:
            adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                client_id="client", client_secret="synthetic", client=Cancelled())
            prepared = await adapter.prepare(request())
            with pytest.raises(asyncio.CancelledError):
                await adapter.create_once(prepared)
        finally:
            logger.remove(sink)
        assert '"reason": "cancelled"' in "".join(messages)
    asyncio.run(run())


@pytest.mark.parametrize("scenario,expected_stage,expected_reason,posts", [
    ("authority_lost", "dispatch", "authority_lost", 0),
    ("expired", "dispatch", "expired_or_backwards_clock", 0),
    ("store_error", "receipt_store", "store_error", 1),
    ("not_saved", "receipt_store", "not_saved", 1),
    ("cancelled", "receipt_store", "cancelled", 1),
    ("unverified", "readback", "unverified", 1),
])
def test_coordinator_distinguishes_post_claim_failure_boundaries(
        scenario, expected_stage, expected_reason, posts):
    from services.scheduling.booking_service import BookingOutcome, BookingService
    from services.scheduling.operation_store import AdmissionStatus, DispatchStatus, OwnershipStatus
    from tests.integration.test_verified_booking_create import (
        Authority, Clock, NOW, SyntheticCalendar, SyntheticSelections,
        VerifiedCreateDatabaseTests, approved,
    )

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield object()

    async def run():
        clock, authority = Clock(), Authority()
        http = SyntheticHttp(read_status=503)
        mutations = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="client", client_secret="synthetic", client=http)
        service = BookingService(Pool(), SyntheticCalendar(), mutations, authority,
                                 clock=clock, selection_loader=SyntheticSelections)
        op = SimpleNamespace(operation_id=uuid4(), state="pending", booking_id=None)

        async def claim(*args, **kwargs):
            if scenario == "authority_lost":
                authority.allowed = False
            elif scenario == "expired":
                clock.now = NOW + timedelta(minutes=5)
            return SimpleNamespace(status=DispatchStatus.GRANTED, fence=1, owner_token=uuid4())

        save = AsyncMock(return_value=scenario == "unverified")
        if scenario == "store_error":
            save.side_effect = RuntimeError("PRIVATE-DATABASE")
        elif scenario == "cancelled":
            save.side_effect = asyncio.CancelledError
        service.store = SimpleNamespace(
            admit=AsyncMock(return_value=SimpleNamespace(status=AdmissionStatus.CREATED, operation=op)),
            claim_dispatch=claim, save_receipt=save,
            inspect_dispatch_owner=AsyncMock(return_value=SimpleNamespace(
                status=OwnershipStatus.CURRENT, lease_until=NOW + timedelta(minutes=1))),
        )
        messages = []
        sink = logger.add(lambda entry: messages.append(str(entry)), format="{message}")
        try:
            if scenario == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await service.create(approved(), VerifiedCreateDatabaseTests._context())
            else:
                result = await service.create(approved(), VerifiedCreateDatabaseTests._context())
                assert result.outcome is BookingOutcome.PENDING
        finally:
            logger.remove(sink)
        evidence = [json.loads(m.split("BookingCreateDiagnostic ", 1)[1])
                    for m in messages if "BookingCreateDiagnostic " in m]
        assert any(e["stage"] == expected_stage and e["reason"] == expected_reason for e in evidence)
        assert sum(m == "POST" and "graph.microsoft.com" in u for m, u, _ in http.calls) == posts
        assert "PRIVATE-" not in "".join(messages)
    asyncio.run(run())
