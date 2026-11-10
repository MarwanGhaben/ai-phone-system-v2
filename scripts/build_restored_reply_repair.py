"""Raise the reply length caps in the exact deployed restored orchestrator.

This is a local source builder, not a server deployment command. Only the six
reply-generation token limits inside ``_get_ai_response`` change; the separate
greeting call elsewhere in the file is deliberately left alone.
"""
import hashlib
import re

BASELINE_SHA = "7ee25e00f5b7a2eba4070f7004896fedfcc49c56080ba6d2d9498eb8fe14ab37"

CAPS = {"80": "220", "60": "160", "50": "120"}
EXPECTED_COUNTS = {"80": 3, "60": 2, "50": 1}
_FUNCTION = "    async def _get_ai_response("
_NEXT_METHOD = re.compile(r"^    (?:async )?def ", re.MULTILINE)
_CAP = re.compile(r"(?P<head>temperature=0\.7,\n[ ]+)max_tokens=(?P<value>80|60|50),\n")


def patch_orchestrator(source: bytes) -> bytes:
    source = source.replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA:
        raise RuntimeError("restored reply source does not match accepted baseline")
    text = source.decode("utf-8")
    if text.count(_FUNCTION) != 1:
        raise RuntimeError("restored reply generator anchor differs")
    start = text.index(_FUNCTION)
    following = _NEXT_METHOD.search(text, start + len(_FUNCTION))
    if following is None:
        raise RuntimeError("restored reply generator end differs")
    end = following.start()
    body = text[start:end]
    counts = {value: 0 for value in CAPS}

    def replace(match):
        value = match.group("value")
        counts[value] += 1
        return match.group("head") + "max_tokens=" + CAPS[value] + ",\n"

    body = _CAP.sub(replace, body)
    if counts != EXPECTED_COUNTS:
        raise RuntimeError("restored reply token caps differ")
    result = text[:start] + body + text[end:]
    compile(result, "restored_reply_repair", "exec")
    return result.encode("utf-8")
