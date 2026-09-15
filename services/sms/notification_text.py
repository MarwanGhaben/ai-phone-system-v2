"""Stable bilingual appointment notices; no model or provider-generated wording."""
from __future__ import annotations

from datetime import datetime

from services.calendar.business_time import BUSINESS_TIMEZONE


def render_notice(kind: str, consultant: str, start: datetime, language: str) -> str:
    if not isinstance(start, datetime) or start.utcoffset() is None:
        raise ValueError('notification time must be aware')
    if not isinstance(consultant, str) or not consultant.strip():
        raise ValueError('consultant required')
    local = start.astimezone(BUSINESS_TIMEZONE)
    when = local.strftime('%A, %B %d, %Y at %I:%M %p %Z')
    date = local.strftime('%Y-%m-%d')
    clock = local.strftime('%I:%M')
    period = '\u0635\u0628\u0627\u062d\u064b\u0627' if local.hour < 12 else '\u0645\u0633\u0627\u0621\u064b'
    if kind == 'removal':
        if language == 'ar':
            return f'\u0645\u0648\u0639\u062f\u0643 \u0645\u0639 {consultant} \u064a\u0648\u0645 {date} \u0627\u0644\u0633\u0627\u0639\u0629 {clock} {period} \u0628\u062a\u0648\u0642\u064a\u062a \u062a\u0648\u0631\u0648\u0646\u062a\u0648 \u0644\u0645 \u064a\u0639\u062f \u0645\u062c\u062f\u0648\u0644\u064b\u0627. \u064a\u0631\u062c\u0649 \u0627\u0644\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 Flexible Accounting \u0644\u062a\u0631\u062a\u064a\u0628 \u0645\u0648\u0639\u062f \u0622\u062e\u0631.'
        return (f'Your appointment with {consultant} on {when} is no longer '
                'scheduled. Please contact Flexible Accounting to arrange another time.')
    if kind == 'confirmation':
        if language == 'ar':
            return f'\u062a\u0645 \u062a\u0623\u0643\u064a\u062f \u0645\u0648\u0639\u062f\u0643 \u0645\u0639 {consultant} \u064a\u0648\u0645 {date} \u0627\u0644\u0633\u0627\u0639\u0629 {clock} {period} \u0628\u062a\u0648\u0642\u064a\u062a \u062a\u0648\u0631\u0648\u0646\u062a\u0648. Flexible Accounting.'
        return f'Your appointment with {consultant} on {when} is confirmed. Flexible Accounting.'
    if kind == 'reminder':
        if language == 'ar':
            return f'\u062a\u0630\u0643\u064a\u0631: \u0645\u0648\u0639\u062f\u0643 \u0645\u0639 {consultant} \u064a\u0648\u0645 {date} \u0627\u0644\u0633\u0627\u0639\u0629 {clock} {period} \u0628\u062a\u0648\u0642\u064a\u062a \u062a\u0648\u0631\u0648\u0646\u062a\u0648. Flexible Accounting.'
        return f'Reminder: Your appointment with {consultant} is on {when}. Flexible Accounting.'
    raise ValueError('unknown notification kind')
