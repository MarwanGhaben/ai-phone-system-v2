"""
=====================================================
AI Voice Platform v2 - Arabic Text Processor
=====================================================
Converts numbers, times, and dates in Arabic text to
Arabic words so TTS can pronounce them naturally.

ElevenLabs TTS reads digits/English words in Arabic text
as garbled nonsense. This processor fixes that by converting
all numbers and time expressions to Arabic words BEFORE
sending to TTS.
"""

import re
from loguru import logger


# =====================================================
# NUMBER TO ARABIC WORD MAPPINGS
# =====================================================

ARABIC_ONES = {
    0: "صفر", 1: "واحد", 2: "اثنين", 3: "ثلاثة", 4: "أربعة",
    5: "خمسة", 6: "ستة", 7: "سبعة", 8: "ثمانية", 9: "تسعة",
    10: "عشرة", 11: "أحد عشر", 12: "اثنا عشر",
    13: "ثلاثة عشر", 14: "أربعة عشر", 15: "خمسة عشر",
    16: "ستة عشر", 17: "سبعة عشر", 18: "ثمانية عشر", 19: "تسعة عشر",
}

ARABIC_TENS = {
    20: "عشرين", 30: "ثلاثين", 40: "أربعين", 50: "خمسين",
}

# Hour names (feminine for ساعة)
ARABIC_HOURS = {
    1: "الواحدة", 2: "الثانية", 3: "الثالثة", 4: "الرابعة",
    5: "الخامسة", 6: "السادسة", 7: "السابعة", 8: "الثامنة",
    9: "التاسعة", 10: "العاشرة", 11: "الحادية عشرة", 12: "الثانية عشرة",
}

# Day names
ENGLISH_TO_ARABIC_DAYS = {
    "monday": "الاثنين",
    "tuesday": "الثلاثاء",
    "wednesday": "الأربعاء",
    "thursday": "الخميس",
    "friday": "الجمعة",
    "saturday": "السبت",
    "sunday": "الأحد",
}

# Month names
ENGLISH_TO_ARABIC_MONTHS = {
    "january": "يناير", "february": "فبراير", "march": "مارس",
    "april": "أبريل", "may": "مايو", "june": "يونيو",
    "july": "يوليو", "august": "أغسطس", "september": "سبتمبر",
    "october": "أكتوبر", "november": "نوفمبر", "december": "ديسمبر",
}

# Ordinal numbers for dates (1-31)
ARABIC_ORDINALS = {
    1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع",
    5: "الخامس", 6: "السادس", 7: "السابع", 8: "الثامن",
    9: "التاسع", 10: "العاشر", 11: "الحادي عشر", 12: "الثاني عشر",
    13: "الثالث عشر", 14: "الرابع عشر", 15: "الخامس عشر",
    16: "السادس عشر", 17: "السابع عشر", 18: "الثامن عشر",
    19: "التاسع عشر", 20: "العشرين", 21: "الحادي والعشرين",
    22: "الثاني والعشرين", 23: "الثالث والعشرين", 24: "الرابع والعشرين",
    25: "الخامس والعشرين", 26: "السادس والعشرين", 27: "السابع والعشرين",
    28: "الثامن والعشرين", 29: "التاسع والعشرين", 30: "الثلاثين",
    31: "الحادي والثلاثين",
}


def _number_to_arabic(n: int) -> str:
    """Convert an integer (0-59) to Arabic words"""
    if n in ARABIC_ONES:
        return ARABIC_ONES[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return ARABIC_TENS.get(tens, str(n))
    return f"{ARABIC_ONES[ones]} و{ARABIC_TENS.get(tens, str(tens))}"


def _time_to_arabic(hour: int, minute: int, am_pm: str = "") -> str:
    """
    Convert a time to natural Arabic speech.

    Examples:
        2:30 PM  → "الثانية والنصف بعد الظهر"
        10:00 AM → "العاشرة صباحاً"
        3:15 PM  → "الثالثة والربع بعد الظهر"
        4:45 PM  → "الخامسة إلا ربع مساءً"
    """
    # Convert 24h to 12h if needed
    if hour > 12:
        hour = hour - 12
        if not am_pm:
            am_pm = "PM"
    elif hour == 0:
        hour = 12
        if not am_pm:
            am_pm = "AM"

    hour_word = ARABIC_HOURS.get(hour, str(hour))

    # Handle minutes
    if minute == 0:
        minute_part = ""
    elif minute == 15:
        minute_part = " والربع"
    elif minute == 30:
        minute_part = " والنصف"
    elif minute == 45:
        # "5 إلا ربع" = quarter to the next hour
        next_hour = hour + 1 if hour < 12 else 1
        hour_word = ARABIC_HOURS.get(next_hour, str(next_hour))
        minute_part = " إلا ربع"
    else:
        minute_word = _number_to_arabic(minute)
        minute_part = f" و{minute_word} دقيقة"

    # AM/PM
    period = ""
    if am_pm:
        am_pm_upper = am_pm.strip().upper()
        if am_pm_upper == "AM":
            period = " صباحاً"
        elif am_pm_upper == "PM":
            period = " مساءً"

    return f"الساعة {hour_word}{minute_part}{period}"


def process_arabic_text(text: str) -> str:
    """
    Process Arabic text to convert numbers and English time/date
    expressions to Arabic words for natural TTS pronunciation.

    Args:
        text: Text that may contain numbers and English words

    Returns:
        Text with numbers converted to Arabic words
    """
    original = text

    # =====================================================
    # 1. Convert full time expressions: "2:30 PM", "10:00 AM"
    # Patterns: HH:MM AM/PM, HH:MM am/pm
    # =====================================================
    def replace_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        am_pm = match.group(3) if match.group(3) else ""
        return _time_to_arabic(hour, minute, am_pm)

    # Match patterns like "2:30 PM", "10:00 AM", "02:30 PM"
    text = re.sub(
        r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?',
        replace_time,
        text
    )

    # =====================================================
    # 2. Convert English day names
    # =====================================================
    for eng, ar in ENGLISH_TO_ARABIC_DAYS.items():
        text = re.sub(rf'\b{eng}\b', ar, text, flags=re.IGNORECASE)

    # =====================================================
    # 3. Convert English month names
    # =====================================================
    for eng, ar in ENGLISH_TO_ARABIC_MONTHS.items():
        text = re.sub(rf'\b{eng}\b', ar, text, flags=re.IGNORECASE)

    # =====================================================
    # 4. Convert date numbers with ordinals (e.g., "02" in dates)
    # Pattern: day number that appears near a month name
    # =====================================================
    def replace_date_number(match):
        num = int(match.group(1))
        if 1 <= num <= 31:
            return ARABIC_ORDINALS.get(num, str(num))
        return match.group(0)

    # Match standalone numbers 01-31 that appear to be dates
    # (near Arabic month names or after "يوم")
    for month_ar in ENGLISH_TO_ARABIC_MONTHS.values():
        # Pattern: "فبراير 02" or "02 فبراير"
        text = re.sub(
            rf'{month_ar}\s+(\d{{1,2}})',
            lambda m: f"{month_ar} {ARABIC_ORDINALS.get(int(m.group(1)), m.group(1))}",
            text
        )
        text = re.sub(
            rf'(\d{{1,2}})\s+{month_ar}',
            lambda m: f"{ARABIC_ORDINALS.get(int(m.group(1)), m.group(1))} {month_ar}",
            text
        )

    # =====================================================
    # 5. Convert remaining standalone numbers
    # =====================================================
    def replace_standalone_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            if 0 <= num <= 59:
                return _number_to_arabic(num)
        except ValueError:
            pass
        return num_str

    # Replace remaining 1-2 digit numbers that weren't already converted
    text = re.sub(r'\b\d{1,2}\b', replace_standalone_number, text)

    # =====================================================
    # 6. Clean up "at" that appears in time expressions
    # =====================================================
    text = re.sub(r'\bat\b', 'في', text, flags=re.IGNORECASE)

    if text != original:
        logger.info(f"Arabic text processor: '{original[:80]}...' → '{text[:80]}...'")

    return text
