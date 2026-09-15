"""Structured Telnyx submission boundary for durable outbox jobs.

An accepted message ID is submission evidence, never a delivery receipt.
Uncertain transport or malformed success is never automatically retried.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


E164 = re.compile(r'^\+[1-9][0-9]{7,14}$')
API_URL = 'https://api.telnyx.com/v2/messages'


@dataclass(frozen=True)
class SubmissionResult:
    state: str
    category: str | None = None
    message_id: str | None = None


async def submit_message(api_key: str, from_number: str, to_number: str,
                         body: str, *, client: object | None = None) -> SubmissionResult:
    if not api_key or not E164.fullmatch(from_number or ''):
        return SubmissionResult('failed', 'configuration')
    if not E164.fullmatch(to_number or ''):
        return SubmissionResult('failed', 'invalid_recipient')
    if not body:
        return SubmissionResult('failed', 'invalid_content')
    owns = client is None
    if client is None:
        import httpx
        client = httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=False)
    try:
        try:
            response = await client.post(
                API_URL,
                headers={'Authorization': 'Bearer ' + api_key,
                         'Content-Type': 'application/json'},
                json={'from': from_number, 'to': to_number, 'text': body})
        except Exception:
            return SubmissionResult('unknown', 'transport_uncertain')
        if response.status_code in (200, 201, 202):
            try:
                payload = response.json()
                identifier = payload['data']['id']
            except (ValueError, TypeError, KeyError):
                return SubmissionResult('unknown', 'malformed_acceptance')
            if not isinstance(identifier, str) or not identifier.strip():
                return SubmissionResult('unknown', 'missing_message_id')
            return SubmissionResult('accepted', message_id=identifier)
        # These validation/auth rejections do not report submission acceptance.
        # Other HTTP statuses are uncertain; no speculative POST retry.
        if response.status_code in (400, 401, 403, 404, 422):
            return SubmissionResult('failed', 'provider_rejected')
        return SubmissionResult('unknown', 'provider_uncertain')
    finally:
        if owns:
            await client.aclose()
