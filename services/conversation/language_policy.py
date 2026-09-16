"""Deterministic policy for explicit caller language requests.

The matcher intentionally recognizes only complete, auditable utterances.  It is
not a language detector and must not infer a preference from arbitrary caller or
assistant text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal


LanguageDecision = Literal["ar", "en"]

_SURROUNDING_PUNCTUATION = ".,!?؟،؛:"

_ARABIC_REQUESTS = frozenset(
    {
        "arabic",
        "arabic please",
        "please speak arabic",
        "speak arabic",
        "speak in arabic",
        "can you speak arabic",
        "switch to arabic",
        "change to arabic",
        "talk arabic",
        "talk in arabic",
        "please talk in arabic",
        "in arabic please",
        "use arabic",
        "respond in arabic",
        "answer in arabic",
        "reply in arabic",
        "want arabic",
        "i want arabic",
        "prefer arabic",
        "i prefer arabic",
        "let's speak arabic",
        "lets speak arabic",
        "let's talk arabic",
        "lets talk arabic",
        "always speak to me in arabic",
        "save arabic as my language",
        "remember arabic",
        "remember i prefer arabic",
        "arabi",
        "arabi please",
        "arabik",
        "arabik please",
        "3arabi",
        "3arabi please",
        "عربي",
        "العربية",
        "بالعربي",
        "بالعربية",
        "عربي من فضلك",
        "العربية من فضلك",
        "بالعربي من فضلك",
        "بالعربية من فضلك",
        "تكلم عربي",
        "تكلم بالعربي",
        "تكلم بالعربية",
        "تحدث بالعربي",
        "تحدث بالعربية",
        "احكي عربي",
        "احكي بالعربي",
        "احكي معي عربي",
        "احكي معي بالعربي",
        "كلمني عربي",
        "كلمني بالعربي",
        "حكي عربي",
        "حكي بالعربي",
        "من فضلك تكلم بالعربي",
        "لو سمحت احكي معي بالعربي",
        "دايما كلمني بالعربي",
        "دايماً كلمني بالعربي",
        "احفظ العربي",
        "احفظ العربية",
        "احفظ اللغة العربية",
    }
)

_ENGLISH_REQUESTS = frozenset(
    {
        "english",
        "english please",
        "please speak english",
        "speak english",
        "speak in english",
        "can you speak english",
        "switch to english",
        "change to english",
        "talk english",
        "talk in english",
        "please talk in english",
        "in english please",
        "use english",
        "respond in english",
        "answer in english",
        "reply in english",
        "want english",
        "i want english",
        "prefer english",
        "i prefer english",
        "let's speak english",
        "lets speak english",
        "let's talk english",
        "lets talk english",
        "always speak to me in english",
        "save english as my language",
        "remember english",
        "remember i prefer english",
        "inglish",
        "inglish please",
        "inglizi",
        "inglizi please",
        "انجليزي",
        "إنجليزي",
        "انكليزي",
        "إنكليزي",
        "انجليش",
        "انجلش",
        "بالانجليزي",
        "بالإنجليزي",
        "بالانكليزي",
        "بالإنكليزي",
        "بالانجليزي من فضلك",
        "بالإنجليزي من فضلك",
        "بالانكليزي من فضلك",
        "بالإنكليزي من فضلك",
        "تكلم انجليزي",
        "تكلم إنجليزي",
        "تكلم انكليزي",
        "تكلم إنكليزي",
        "تكلم بالانجليزي",
        "تكلم بالإنجليزي",
        "تحدث بالانجليزي",
        "تحدث بالإنجليزي",
        "احكي انجليزي",
        "احكي إنجليزي",
        "احكي معي بالانجليزي",
        "احكي معي بالإنجليزي",
        "كلمني انجليزي",
        "كلمني إنجليزي",
        "كلمني بالانجليزي",
        "كلمني بالإنجليزي",
        "من فضلك تكلم بالانجليزي",
        "من فضلك تكلم بالإنجليزي",
        "احفظ الانجليزي",
        "احفظ الإنجليزية",
        "احفظ اللغة الإنجليزية",
    }
)


def _normalize_utterance(utterance: str) -> str:
    normalized = unicodedata.normalize("NFKC", utterance).casefold().strip()
    normalized = normalized.strip(_SURROUNDING_PUNCTUATION).strip()
    return re.sub(r"\s+", " ", normalized)


def explicit_language_request(utterance: str) -> LanguageDecision | None:
    """Return the explicitly requested language for a complete utterance."""

    if not isinstance(utterance, str):
        return None

    normalized = _normalize_utterance(utterance)
    if normalized in _ARABIC_REQUESTS:
        return "ar"
    if normalized in _ENGLISH_REQUESTS:
        return "en"
    return None
