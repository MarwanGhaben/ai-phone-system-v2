"""Independent offline T049-A policy and notification regressions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class RemovalPolicyTests(unittest.TestCase):
    def test_two_spaced_absences_require_live_enrollment(self):
        from services.scheduling.removal_reconciliation import Evidence, advance_evidence
        t = datetime(2026, 9, 15, 15, tzinfo=timezone.utc)
        old = Evidence()
        missing, ready = advance_evidence(old, 'unavailable', t)
        self.assertEqual((missing.missing_count, ready), (0, False))
        enrolled, ready = advance_evidence(old, 'present', t)
        self.assertTrue(enrolled.enrolled)
        first, ready = advance_evidence(enrolled, 'unavailable', t + timedelta(seconds=30))
        self.assertEqual((first.missing_count, ready), (1, False))
        second, ready = advance_evidence(first, 'unavailable', t + timedelta(seconds=90))
        self.assertEqual((second.missing_count, ready), (2, True))

    def test_error_present_and_bad_spacing_reset_missing_sequence(self):
        from services.scheduling.removal_reconciliation import Evidence, advance_evidence
        t = datetime(2026, 9, 15, 15, tzinfo=timezone.utc)
        for interruption in ('present', 'changed', 'check_failed'):
            first, _ = advance_evidence(Evidence(enrolled=True), 'unavailable', t)
            reset, _ = advance_evidence(first, interruption, t + timedelta(seconds=60))
            second, ready = advance_evidence(reset, 'unavailable', t + timedelta(seconds=120))
            self.assertEqual((second.missing_count, ready), (1, False))
        for gap in (timedelta(seconds=59), timedelta(minutes=11)):
            first, _ = advance_evidence(Evidence(enrolled=True), 'unavailable', t)
            second, ready = advance_evidence(first, 'unavailable', t + gap)
            self.assertEqual((second.missing_count, ready), (1, False))


class NotificationTextTests(unittest.TestCase):
    def setUp(self):
        try:
            ZoneInfo('America/Toronto')
        except ZoneInfoNotFoundError:
            self.skipTest('this CLI lacks tzdata; normal app image must exercise Toronto')

    def test_removal_wording_avoids_unproved_cause_in_two_languages(self):
        from services.sms.notification_text import render_notice
        summer = datetime(2026, 9, 15, 15, tzinfo=timezone.utc)
        winter = datetime(2027, 1, 15, 16, tzinfo=timezone.utc)
        en = render_notice('removal', 'Hussam', summer, 'en')
        ar = render_notice('removal', 'Hussam', winter, 'ar')
        self.assertIn('no longer scheduled', en)
        self.assertIn('11:00', en)
        self.assertIn('Hussam', ar)
        self.assertIn('11:00', ar)
        self.assertIn('\u0644\u0645 \u064a\u0639\u062f \u0645\u062c\u062f\u0648\u0644', ar)
        self.assertNotIn('????', ar)
        self.assertNotIn('cancelled by', en)
        self.assertNotIn('delivered', en)

    def test_invalid_time_fails_and_unknown_language_uses_english(self):
        from services.sms.notification_text import render_notice
        t = datetime(2026, 9, 15, 15, tzinfo=timezone.utc)
        self.assertEqual(render_notice('removal', 'Hussam', t, 'unknown'),
                         render_notice('removal', 'Hussam', t, 'en'))
        with self.assertRaises(ValueError):
            render_notice('removal', 'Hussam', t.replace(tzinfo=None), 'en')


class _Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}

    def json(self):
        return self.body


class _Graph:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        return self.responses.pop(0)


class InventoryTests(unittest.IsolatedAsyncioTestCase):
    async def _inventory(self, responses):
        from services.calendar.booking_readback import BookingReadbackClient
        settings = SimpleNamespace(ms_bookings_tenant_id='tenant',
                                   ms_bookings_business_id='business@example.invalid')
        graph = _Graph(responses)
        client = BookingReadbackClient(settings, client=graph)
        client._token, client._token_until = 'synthetic', float('inf')
        result = await client.inventory_absent('target', 'tenant', 'business@example.invalid')
        return result, graph.calls

    async def test_complete_inventory_absence_and_later_page_match(self):
        from services.calendar.booking_readback import GRAPH_HOST
        base = GRAPH_HOST + '/v1.0/solutions/bookingBusinesses/business%40example.invalid/appointments'
        business = _Response(body={'id': 'business@example.invalid'})
        first = _Response(body={'value': [{'id': 'other'}],
                                '@odata.nextLink': base + '?%24skiptoken=opaque'})
        absent, calls = await self._inventory([
            business, first, _Response(body={'value': [{'id': 'another'}]})])
        self.assertIs(absent.absent, True)
        self.assertEqual(len(calls), 3)
        present, _ = await self._inventory([
            business, first, _Response(body={'value': [{'id': 'target'}]})])
        self.assertIs(present.absent, False)

    async def test_wrong_business_bad_pages_and_unsafe_continuations_hold(self):
        business = _Response(body={'id': 'business@example.invalid'})
        for responses in (
            [_Response(body={'id': 'wrong'})],
            [business, _Response(body={'value': [{'id': None}]})],
            [business, _Response(body={'value': [],
                                       '@odata.nextLink': 'https://evil.invalid/x'})],
            [business, _Response(body={'value': [],
                                       '@odata.nextLink': 'https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/other/appointments?$skiptoken=x'})],
        ):
            with self.subTest(responses=responses):
                result, _ = await self._inventory(responses)
                self.assertIsNone(result.absent)

    async def test_inventory_loop_duplicate_id_and_page_budget_never_prove_absence(self):
        from services.calendar.booking_readback import GRAPH_HOST
        base = (GRAPH_HOST
                + '/v1.0/solutions/bookingBusinesses/business%40example.invalid/appointments')
        business = _Response(body={'id': 'business@example.invalid'})
        for pages in (
            [_Response(body={'value': [], '@odata.nextLink': base + '?$skiptoken=one'}),
             _Response(body={'value': [], '@odata.nextLink': base + '?$skiptoken=one'})],
            [_Response(body={'value': [{'id': 'same'}],
                             '@odata.nextLink': base + '?$skiptoken=two'}),
             _Response(body={'value': [{'id': 'same'}]})],
            [_Response(body={'value': [],
                             '@odata.nextLink': base + f'?$skiptoken={n + 1}'})
             for n in range(20)],
            [_Response(body={'value': [{'id': f'id-{n}'} for n in range(2001)]})],
        ):
            with self.subTest(pages=len(pages)):
                result, _ = await self._inventory([business, *pages])
                self.assertIsNone(result.absent)

    async def test_inventory_throttle_and_redirect_are_inconclusive(self):
        business = _Response(body={'id': 'business@example.invalid'})
        throttled, _ = await self._inventory([
            business, _Response(429, headers={'Retry-After': '120'})])
        self.assertEqual((throttled.absent, throttled.error_category,
                          throttled.retry_after, throttled.stop_batch),
                         (None, 'throttled', 120, True))
        redirect, _ = await self._inventory([
            business, _Response(302, headers={'Location': 'https://evil.invalid'})])
        self.assertIsNone(redirect.absent)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_terminates_session_and_releases_admission(self):
        import asyncio
        from services.calendar.provider_admission import admitted

        mutex = asyncio.Lock()
        first_entered, release_first, second_waiting = (
            asyncio.Event(), asyncio.Event(), asyncio.Event())

        class Conn:
            def __init__(self, waiting=None):
                self.waiting, self.closed = waiting, False

            async def fetchval(self, query, *args):
                if 'pg_advisory_unlock' in query:
                    mutex.release()
                    return True
                if 'pg_advisory_lock' in query:
                    if self.waiting is not None:
                        self.waiting.set()
                    await mutex.acquire()
                    return None
                if 'next_request_at > CURRENT_TIMESTAMP' in query:
                    return False
                raise AssertionError(query)

            def is_closed(self):
                return self.closed

            def terminate(self):
                self.closed = True

        async def first_reader():
            async with admitted(Conn(), object()) as allowed:
                self.assertTrue(allowed)
                first_entered.set()
                await release_first.wait()

        async def second_reader(conn):
            async with admitted(conn, object()):
                self.fail('cancelled waiter was admitted')

        first = asyncio.create_task(first_reader())
        try:
            await asyncio.wait_for(first_entered.wait(), 2)
            second_conn = Conn(second_waiting)
            second = asyncio.create_task(second_reader(second_conn))
            await asyncio.wait_for(second_waiting.wait(), 2)
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            self.assertTrue(second_conn.closed)
        finally:
            release_first.set()
            await asyncio.wait_for(first, 2)

        async with admitted(Conn(), object()) as allowed:
            self.assertTrue(allowed)

    async def test_shared_pause_blocks_next_fake_read_and_inventory_page(self):
        from services.calendar.booking_readback import (
            BookingReadbackClient, ProviderResult)
        from services.calendar.provider_admission import (
            inventory_absent, observe)

        class Conn:
            def __init__(self): self.paused,self.closed = False,False
            async def fetchval(self, query, *args):
                if 'pg_advisory_unlock' in query: return True
                if 'next_request_at > CURRENT_TIMESTAMP' in query: return self.paused
                if 'clock_timestamp()' in query: return datetime.now(timezone.utc)
                return None
            async def execute(self, query, *args): self.paused = True
            def is_closed(self): return self.closed
            def terminate(self): self.closed = True

        class Graph:
            def __init__(self): self.calls = 0
            async def observe(self, provider_id, start):
                self.calls += 1
                return ProviderResult('check_failed','throttled',retry_after=3600,
                                      stop_batch=True)

        conn,graph = Conn(),Graph()
        first,_ = await observe(conn,graph,'synthetic-id',datetime.now(timezone.utc))
        denied,started = await observe(
            conn,graph,'synthetic-id',datetime.now(timezone.utc))
        self.assertTrue(first.stop_batch)
        self.assertEqual((denied.error_category,started,graph.calls),
                         ('throttled',None,1))

        conn.paused = False
        class HTTP:
            def __init__(self): self.calls = 0
            async def get(self, url, **kwargs):
                self.calls += 1
                conn.paused = True
                return _Response(body={'id': 'business@example.invalid'})

        http = HTTP()
        settings = SimpleNamespace(ms_bookings_tenant_id='tenant',
                                   ms_bookings_business_id='business@example.invalid')
        reader = BookingReadbackClient(settings,client=http)
        reader._token,reader._token_until = 'synthetic-token',float('inf')
        result = await inventory_absent(
            conn,reader,'synthetic-id','tenant','business@example.invalid')
        self.assertIsNone(result.absent)
        self.assertEqual(result.error_category,'throttled')
        self.assertEqual(http.calls,1)


class _Sms:
    def __init__(self, response):
        self.response, self.calls = response, []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_message_id_is_submission_acceptance_not_delivery(self):
        from services.sms.telnyx_submission import submit_message
        sms = _Sms(_Response(202, {'data': {'id': 'synthetic-message'}}))
        result = await submit_message('secret', '+12025550100', '+12025550101',
                                      'Synthetic notice', client=sms)
        self.assertEqual((result.state, result.message_id), ('accepted', 'synthetic-message'))
        self.assertEqual(len(sms.calls), 1)

    async def test_uncertain_response_never_becomes_success_or_safe_retry(self):
        from services.sms.telnyx_submission import submit_message
        for response in (_Response(202, {'data': {}}), TimeoutError('secret'),
                         _Response(503, {})):
            result = await submit_message('secret', '+12025550100', '+12025550101',
                                          'Synthetic notice', client=_Sms(response))
            self.assertEqual(result.state, 'unknown')
            self.assertNotIn('secret', repr(result))
        invalid = _Sms(_Response(202, {'data': {'id': 'id'}}))
        result = await submit_message('secret', '+12025550100', 'private',
                                      'Synthetic notice', client=invalid)
        self.assertEqual(result.state, 'failed')
        self.assertEqual(invalid.calls, [])


if __name__ == '__main__':
    unittest.main()
