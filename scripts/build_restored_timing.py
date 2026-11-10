"""Add per-reply timing measurement to the exact deployed restored orchestrator.

This is a local source builder, not a server deployment command. The patch is
insert-only: no existing line is changed, removed or reordered. Every inserted
statement is wrapped so that an exception inside the measurement cannot propagate
into the call flow, and nothing new is awaited. The measurement logic lives in
services/conversation/turn_timing.py.
"""
import difflib
import hashlib
import re

BASELINE_SHA = "5ef96383a930d4fa38e8943bac7abd0afc47fd965b639ac42b6270b6410b7561"
PATCHED_SHA = "f8ee6d927ce287425d9ce46a72ec154322e9a6f983fe4381cb5ce23331703f43"

IMPORT_BLOCK = (
    "try:\n"
    "    from services.conversation import turn_timing as _tt_mod\n"
    "except Exception:  # measurement is optional and never affects calls\n"
    "    _tt_mod = None\n"
)
FINISH_BODY = (
    "if _tt is not None:\n"
    "    _line = _tt.finish()\n"
    "    if _line:\n"
    "        logger.info(_line)\n"
)
TAKE_BLOCK = (
    "_tt = None\n"
    "try:\n"
    "    _tt = _tt_mod.take(self._conversations.get(call_sid))\n"
    "except Exception:\n"
    "    _tt = None\n"
)


def _guard(indent, body):
    lines = body.rstrip("\n").split("\n")
    out = [indent + "try:"]
    out += [indent + "    " + line for line in lines]
    out += [indent + "except Exception:", indent + "    pass"]
    return "\n".join(out) + "\n"


def _indented(indent, block):
    return "".join(indent + line + "\n" for line in block.rstrip("\n").split("\n"))


# (name, regex matching whole lines, where, builder(indent) -> text, expected count)
# where: "before" the match, "after" the match, both relative to the matched text.
def _specs():
    mod = "_tt_mod."
    return [
        ("import", r"^()from services\.conversation\.language_policy import explicit_language_request\n",
         "after", lambda i: IMPORT_BLOCK, 1),
        ("note-received", r"^( *)await self\.process_transcript\(\n +call_sid,\n +result\.text,\n",
         "before", lambda i: _guard(i, mod + "note_received(context, result)"), 1),
        ("begin", r'^( *)logger\.info\(f"Orchestrator: User said: \{transcript\}"\)\n',
         "before", lambda i: _guard(i, mod + "begin(context)"), 1),
        ("ai-start", r"^( *)response = await self\._get_ai_response\(call_sid, transcript\)\n",
         "before", lambda i: _guard(i, mod + "mark(context, 'ai_start')"), 1),
        ("ai-end", r"^( *)response = await self\._get_ai_response\(call_sid, transcript\)\n",
         "after", lambda i: _guard(i, mod + "mark(context, 'ai_end')"), 1),
        ("arm", r"^( *)# Stream response to caller \(this waits for audio to complete\)\n",
         "before", lambda i: _guard(i, mod + "arm(context)"), 1),
        ("llm-first-start", r"^( *)llm_response = await self\.llm\.chat_with_tools\(request\)\n",
         "before", lambda i: _guard(i, mod + "llm_start(context)"), 1),
        ("llm-first-done", r"^( *)llm_response = await self\.llm\.chat_with_tools\(request\)\n",
         "after", lambda i: _guard(i, mod + "llm_first_done(context)"), 1),
        ("tool", r"^( *)for tool_call in llm_response\.tool_calls:\n",
         "after", lambda i: _guard(i + "    ", mod + "tool(context)"), 1),
        ("llm-follow-up", r"^( *)async for chunk in self\.llm\.chat_stream\(follow_up_request\):\n",
         "before", lambda i: _guard(i, mod + "llm_start(context)"), 3),
        ("llm-fallback", r"^( *)async for chunk in self\.llm\.chat_stream\(request\):\n",
         "before", lambda i: _guard(i, mod + "llm_start(context)"), 1),
        ("llm-farewell", r"^( *)farewell_response = await self\.llm\.chat\(farewell_request\)\n",
         "before", lambda i: _guard(i, mod + "llm_start(context)"), 1),
        ("speak-take", r'^( *)logger\.info\("Orchestrator: _speak_to_caller START"\)\n',
         "after", lambda i: _indented(i, TAKE_BLOCK), 1),
        ("tts-request", r"^( *)audio_stream = self\.tts\.create_stream\(request\)\n",
         "after", lambda i: _guard(i, "if _tt is not None:\n    _tt.request_created(audio_stream)"), 1),
        ("streamed", r"^( *)stream_result = await twilio_handler\.stream_audio_chunks\(\n"
                     r" +owner\.playback, owner\.tts_stream\n +\)\n",
         "after", lambda i: _guard(i, "if _tt is not None:\n    _tt.set_bytes(stream_result.bytes_sent)"), 1),
        ("speak-finish", r"^( *)finally:\n(?= +if owner is not None:\n +owner\.tts_stream\.cancel\(\)\n)",
         "after", lambda i: _guard(i + "    ", FINISH_BODY), 1),
    ]


def inserted_blocks():
    """The exact insertions, for documentation and tests."""
    return [(name, where, count) for name, _p, where, _b, count in _specs()]


def patch_orchestrator(source: bytes) -> bytes:
    source = source.replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA:
        raise RuntimeError("restored timing source does not match accepted baseline")
    text = source.decode("utf-8")
    for name, pattern, where, build, expected in _specs():
        regex = re.compile(pattern, re.MULTILINE)
        matches = list(regex.finditer(text))
        if len(matches) != expected:
            raise RuntimeError("restored timing anchor differs: " + name)
        for match in reversed(matches):
            block = build(match.group(1))
            at = match.start() if where == "before" else match.end()
            text = text[:at] + block + text[at:]
    result = text
    old = source.decode("utf-8").split("\n")
    new = result.split("\n")
    for tag, *_ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag not in ("equal", "insert"):
            raise RuntimeError("restored timing patch is not insert-only")
    compile(result, "restored_timing", "exec")
    data = result.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != PATCHED_SHA:
        raise RuntimeError("restored timing result does not match the reviewed hash")
    return data
