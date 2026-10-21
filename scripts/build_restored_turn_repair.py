"""Construct the caller-admission repair from the exact restored runtime source.

This is a local source builder, not a server deployment command. The parked
phase-five orchestrator is deliberately not used as the application baseline.
"""
import hashlib

BASELINE_SHA = "bc8afe8c245b552a7e40cedaef5809497440912948ee2a8ae6e543b8b5082aac"


def patch_orchestrator(source: bytes) -> bytes:
    source = source.replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA:
        raise ValueError("restored caller source does not match accepted baseline")
    text = source.decode("utf-8")
    anchor = "                        await self.process_transcript(\n"
    replacement = (
        "                        from services.conversation.restored_turn_admission import admit_final\n"
        "                        if not await admit_final(self, call_sid, context, call_stt, result):\n"
        "                            if self._conversations.get(call_sid) is not context:\n"
        "                                return\n"
        "                            continue\n"
        + anchor
    )
    if text.count(anchor) != 1:
        raise ValueError("restored transcript consumer anchor differs")
    return text.replace(anchor, replacement).encode("utf-8")
