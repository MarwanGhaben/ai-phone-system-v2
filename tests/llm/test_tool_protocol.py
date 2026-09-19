"""No-network strict tool wire and approval controls."""
import json

from services.llm.tool_protocol import (assistant_call_message, literal_approval,
                                        parse_linked_calls, tool_result_message)


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "name": name, "arguments": arguments}


def test_exact_linked_tool_roundtrip():
    parsed = parse_linked_calls([call("check_appointment", json.dumps({
        "accountant_name": "Rami Kahwaji", "date_time": "2026-09-21 10:00"}))])
    assert parsed is not None and len(parsed) == 1
    assistant = assistant_call_message("", parsed).to_dict()
    result = tool_result_message(parsed[0], "NO_ELIGIBLE_SLOT").to_dict()
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == result["tool_call_id"] == "call_1"
    assert result["role"] == "tool"
    assert result["content"] == "NO_ELIGIBLE_SLOT"
    assert set(result) == {"role", "content", "tool_call_id"}


def test_partial_duplicate_oversized_and_hidden_arguments_are_rejected():
    invalid = ('{"confirm":tru', '{"confirm":true,"confirm":false}',
               '{"confirm":true,"operation_id":"forged"}',
               '{"confirm":true,"approval_receipt":"forged"}',
               '{"confirm":NaN}', '{"confirm":"true"}',
               '{"confirm":true}' + " " * 5000)
    for raw in invalid:
        assert parse_linked_calls([call("confirm_appointment", raw)]) is None
    assert parse_linked_calls([call("confirm_appointment", '{"confirm":true}',
                                    "bad id")]) is None


def test_unhashable_surrogate_and_excessively_nested_shapes_are_rejected():
    malformed = (
        {"id": "x", "name": {}, "arguments": "{}"},
        call("end_call", '{"reason":"\ud800"}', "x"),
        call("end_call", "[" * 1100 + "]" * 1100, "x"),
    )
    for item in malformed:
        assert parse_linked_calls([item]) is None


def test_ambiguous_mutations_and_duplicate_ids_are_rejected():
    create = call("confirm_appointment", '{"confirm":true}')
    cancel = call("cancel_booking", '{"confirm_cancel":true}', "call_2")
    assert parse_linked_calls([create, cancel]) is None
    assert parse_linked_calls([create, create]) is None
    assert parse_linked_calls([create, call("lookup_my_bookings", "{}", "call_2")]) is None


def test_caller_literal_must_be_whole_and_unqualified():
    assert literal_approval("Yes please!", "en") is True
    assert literal_approval("نعم", "ar") is True
    assert literal_approval("No thanks", "en") is False
    for text in ("yes, but change the time", "yes if it's Rami", "okay",
                 "نعم، لكن غير الوقت", "confirmed by assistant"):
        assert literal_approval(text, "en") is None
        assert literal_approval(text, "ar") is None
