"""Strict, linked tool arguments for enabled phone booking conversations."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from services.llm.llm_base import LLMRole, Message


_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SCHEMA = {
    "search_appointments": ({"accountant_name": str, "date": str}, {"accountant_name", "date"}),
    "check_appointment": ({"accountant_name": str, "date_time": str,
                           "customer_name": str, "customer_email": str,
                           "client_type": str}, {"accountant_name", "date_time"}),
    "confirm_appointment": ({"confirm": bool}, {"confirm"}),
    "register_caller_name": ({"caller_name": str}, {"caller_name"}),
    "save_language_preference": ({"language": str}, {"language"}),
    "transfer_to_human": ({"reason": str}, set()),
    "end_call": ({"reason": str}, set()),
    "lookup_my_bookings": ({}, set()),
    "cancel_booking": ({"confirm_cancel": bool, "appointment_number": int},
                       {"confirm_cancel"}),
}
_MUTATIONS = frozenset({"confirm_appointment", "cancel_booking"})


@dataclass(frozen=True, slots=True)
class LinkedCall:
    call_id: str
    name: str
    arguments: dict
    raw_arguments: str


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("invalid constant")


def parse_linked_calls(calls: object) -> tuple[LinkedCall, ...] | None:
    """Reject partial, oversized, duplicate, extra or mixed mutation calls."""
    if type(calls) is not list or len(calls) > 4:
        return None
    parsed = []
    seen = set()
    mutation_count = 0
    for call in calls:
        if type(call) is not dict:
            return None
        call_id, name, raw = (call.get("id"), call.get("name"),
                              call.get("arguments"))
        if (type(call_id) is not str or _ID.fullmatch(call_id) is None
                or call_id in seen or type(name) is not str or name not in _SCHEMA
                or type(raw) is not str):
            return None
        try:
            if len(raw.encode("utf-8")) > 4096:
                return None
            arguments = json.loads(raw, object_pairs_hook=_pairs,
                                   parse_constant=_constant)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            return None
        allowed, required = _SCHEMA[name]
        if (type(arguments) is not dict or not required.issubset(arguments)
                or not arguments.keys() <= allowed.keys()
                or any(type(value) is not allowed[key]
                       for key, value in arguments.items())
                or any(type(value) is str and (len(value) > 255
                        or (not value.strip() and key != "customer_email"))
                       for key, value in arguments.items())):
            return None
        if name == "cancel_booking" and "appointment_number" in arguments:
            if arguments["appointment_number"] < 1:
                return None
        if name == "save_language_preference" and arguments["language"] not in ("en", "ar"):
            return None
        if name == "check_appointment" and arguments.get("client_type", "individual") not in ("individual", "corporate"):
            return None
        seen.add(call_id)
        mutation_count += name in _MUTATIONS
        parsed.append(LinkedCall(call_id, name, arguments, raw))
    if mutation_count > 1 or (mutation_count and len(parsed) > 1):
        return None
    return tuple(parsed)


def assistant_call_message(content: str, calls: tuple[LinkedCall, ...]) -> Message:
    return Message(LLMRole.ASSISTANT, content, {
        "tool_calls": [{"id": call.call_id, "type": "function",
                        "function": {"name": call.name, "arguments": call.raw_arguments}}
                       for call in calls]})


def tool_result_message(call: LinkedCall, result: str) -> Message:
    return Message(LLMRole.TOOL, result, {"tool_call_id": call.call_id, "name": call.name})


def literal_approval(text: object, language: str) -> bool | None:
    """Only a whole, unqualified caller answer can express approval or refusal."""
    if type(text) is not str or len(text) > 128 or language not in ("en", "ar"):
        return None
    value = " ".join(text.casefold().strip(" \t\r\n.!?؟،").split())
    if language == "ar":
        if value in ("نعم", "أؤكد", "نعم أؤكد", "اي نعم", "أوافق"):
            return True
        if value in ("لا", "لا أؤكد", "لا أوافق"):
            return False
    else:
        if value in ("yes", "yes please", "i confirm", "confirm", "i agree"):
            return True
        if value in ("no", "no thanks", "i do not confirm", "i don't confirm"):
            return False
    return None


def caller_approval(text: object, language: str) -> bool | None:
    """An unqualified yes/no need not switch the conversation language."""
    if language not in ("ar", "en"):
        return None
    decision = literal_approval(text, language)
    return (literal_approval(text, "en" if language == "ar" else "ar")
            if decision is None else decision)
