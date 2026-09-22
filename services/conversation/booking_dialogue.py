"""Caller-owned day and language constraints for the verified booking dialogue.

This is a bounded recognizer, not a general date parser. Conflicting dates ask
for clarification; model tool arguments never override a recognized caller day.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
import unicodedata
from zoneinfo import ZoneInfo

from services.conversation.language_policy import explicit_language_request
from services.scheduling import spoken_arabic

TORONTO = ZoneInfo("America/Toronto")
_WEEKDAYS = (
    ("monday", "الاثنين", "الاتنين"), ("tuesday", "الثلاثاء", "الثلاثا"),
    ("wednesday", "الاربعاء", "الاربعا"), ("thursday", "الخميس"),
    ("friday", "الجمعة"), ("saturday", "السبت"), ("sunday", "الاحد"),
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn" and c != "ـ")
    return text.translate(str.maketrans("أإآى٠١٢٣٤٥٦٧٨٩", "اااي0123456789"))


@dataclass(frozen=True)
class CallerDay:
    day: date | None = None
    ambiguous: bool = False


def requested_day(text: str, now: datetime) -> CallerDay:
    today = now.astimezone(TORONTO).date()
    value = normalize(text)
    # Do not discard an unsupported relative qualifier or ambiguous numeric date.
    if re.search(
        r"\blast (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
        r"(?<!\w)(?:الماضي|الفات)(?!\w)|\d\s*[/]\s*\d", value):
        return CallerDay(ambiguous=True)
    days = set()
    for iso in re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", value):
        try:
            days.add(date.fromisoformat(iso))
        except ValueError:
            return CallerDay(ambiguous=True)
    months = ("january يناير", "february فبراير", "march مارس", "april ابريل", "may مايو",
              "june يونيو", "july يوليو", "august اغسطس", "september سبتمبر", "october اكتوبر",
              "november نوفمبر", "december ديسمبر")
    for month, labels in enumerate(months, 1):
        for label in labels.split():
            match = re.search(r"(?<!\w)" + label + r"(?!\w)", value)
            if match is None or (label == "may" and re.search(r"\bmay i\b", value)):
                continue
            before, after = value[:match.start()], value[match.end():]
            numeric = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s*(?:من\s*)?$", before)
            numeric = numeric or re.match(r"\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d)", after)
            day_number = int(numeric[1]) if numeric else None
            if day_number is None:
                for day in range(1, 32):
                    ordinal = normalize(spoken_arabic._DATES[day])
                    cardinal = normalize(spoken_arabic.number(day))
                    for words in {ordinal, cardinal, ordinal.replace("ون", "ين"), cardinal.replace("ون", "ين")}:
                        if re.search(r"(?<!\w)" + re.escape(words) + r"\s*(?:من\s*)?$", before):
                            day_number = day
            years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", value))
            if day_number is None or len(years) > 1:
                return CallerDay(ambiguous=True)
            try:
                days.add(date(int(next(iter(years))) if years else today.year, month, day_number))
            except ValueError:
                return CallerDay(ambiguous=True)
    for weekday, names in enumerate(_WEEKDAYS):
        variants = (*names, *(prefix + name for name in names[1:] for prefix in ("ب", "و")),
                    *("ل" + name[1:] for name in names[1:] if name.startswith("ال")))
        if any(re.search(r"(?<!\w)" + name + r"(?!\w)", value) for name in variants):
            if re.search(r"\b(?:not|except)\b|(?<!\w)(?:مش|مو|غير)(?!\w)", value):
                return CallerDay(ambiguous=True)
            if days:
                if any(day.weekday() != weekday for day in days):
                    return CallerDay(ambiguous=True)
            else:
                days.add(today + timedelta(days=(weekday - today.weekday()) % 7))
    if re.search(r"\bnext week\b|الاسبوع (?:القادم|الجاي|المقبل)", value):
        # A weekday plus 'next week' can have two common interpretations.
        return CallerDay(ambiguous=True)
    after = r"day after tomorrow|بعد (?:بكرة|بكرا|غد|غدا)"
    if re.search(after, value):
        days.add(today + timedelta(days=2))
        value = re.sub(after, "", value)
    if re.search(r"\btomorrow\b|(?<!\w)(?:بكرة|بكرا|غدا|غد)(?!\w)", value):
        days.add(today + timedelta(days=1))
    if re.search(r"\btoday\b|(?<!\w)اليوم(?!\w)", value):
        days.add(today)
    if len(days) > 1:
        return CallerDay(ambiguous=True)
    return CallerDay(next(iter(days)) if days else None)


def has_clock(text: str) -> bool:
    value = re.sub(r"\d{4}-\d{2}-\d{2}", "", normalize(text))
    return bool(re.search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|"
        r"\bat\s+(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|noon|midnight)\b|"
        r"\b(?:الساعة|ساعه|ساعة)\s+(?:\d{1,2}|واحدة|واحده|اثنين|ثلاثة|اربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|عشره)\b|"
        r"\b\d{1,2}:\d{2}\b|(?<!\w)(?:العاشرة|الحادية عشرة|الثانية عشرة|الواحدة|الثانية|الثالثة|الرابعة|الخامسة)(?!\w)",
        value))


def turn_language(text: str, reported: str | None, current: str) -> str:
    explicit = explicit_language_request(text)
    if explicit:
        return explicit
    if re.search(r"[\u0621-\u064a]", text):
        return "ar"
    # Names, yes/no, times and email spellings are not a language switch.
    words = re.findall(r"[a-z]+", text.casefold())
    # Canonical STT commits can omit detected language or carry the configured
    # fallback. Clear caller text must not require a provider 'en' label.
    if (len(words) >= 3
            and (set(words) & {"i", "what", "when", "where", "can", "could", "please", "would", "how"}
                 or re.search(r"\bmy (?:name|email|phone number) is\b", text, re.I))):
        return "en"
    return current if current in ("ar", "en") else "en"


def plain_reply_ok(text: str, language: str) -> bool:
    if language == "en":
        arabic = len(re.findall(r"[\u0621-\u064a]", text))
        latin = len(re.findall(r"[a-z]", text, re.I))
        if arabic > latin:
            return False
    if language == "ar" and not re.search(r"[\u0621-\u064a]", text):
        return False
    if language == "ar" and re.search(r"\b(?:staff member|appointment|toronto time|am|pm)\b", text, re.I):
        return False
    # No phantom search/transfer when the model returned no tool call.
    return not re.search(
        r"\b(?:i(?:'ll| will| am going to)|let me)\s+(?:check|search|look|transfer|connect)|"
        r"(?:سوف |س)(?:اتحقق|ابحث|احول)|خليني (?:اشوف|اتحقق|ابحث)", normalize(text))
