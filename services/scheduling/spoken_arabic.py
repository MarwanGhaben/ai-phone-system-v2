"""Deterministic spoken booking facts; never selects or changes an appointment."""
from datetime import datetime


_ONES = ("صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة")
_TEENS = ("عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر",
          "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر")
_TENS = ("", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون")
_HUNDREDS = ("", "مئة", "مئتان", "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة")
_HOURS = ("الثانية عشرة", "الواحدة", "الثانية", "الثالثة", "الرابعة", "الخامسة",
          "السادسة", "السابعة", "الثامنة", "التاسعة", "العاشرة", "الحادية عشرة")
_DAYS = ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد")
_MONTHS = ("", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
           "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
_DATES = ("", "الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع",
          "الثامن", "التاسع", "العاشر", "الحادي عشر", "الثاني عشر", "الثالث عشر",
          "الرابع عشر", "الخامس عشر", "السادس عشر", "السابع عشر", "الثامن عشر",
          "التاسع عشر", "العشرون", "الحادي والعشرون", "الثاني والعشرون", "الثالث والعشرون",
          "الرابع والعشرون", "الخامس والعشرون", "السادس والعشرون", "السابع والعشرون",
          "الثامن والعشرون", "التاسع والعشرون", "الثلاثون", "الحادي والثلاثون")

# Display-only aliases for the configured consultants. Unknown names stay intact.
_CONSULTANTS = {"Hussam Saadaldin": "حسام سعد الدين", "Rami Kahwaji": "رامي قهوجي",
                "Abdul ElFarra": "عبدول الفرا", "Abdul": "عبدول"}


def number(value: int) -> str:
    if type(value) is not int or not 0 <= value <= 9999:
        raise ValueError("unsupported spoken number")
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value - 10]
    if value < 100:
        tens, ones = divmod(value, 10)
        return (_ONES[ones] + " و" if ones else "") + _TENS[tens]
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return _HUNDREDS[hundreds] + (" و" + number(rest) if rest else "")
    thousands, rest = divmod(value, 1000)
    prefix = "ألف" if thousands == 1 else "ألفين" if thousands == 2 else _ONES[thousands] + " آلاف"
    return prefix + (" و" + number(rest) if rest else "")


def consultant(name: str) -> str:
    return _CONSULTANTS.get(name, name)


def service(name: str) -> str:
    return {"appointment": "موعد", "30-min meeting": "موعد", "اجتماع 30 دقيقة": "اجتماع لمدة نصف ساعة"}.get(name, name)


def date(value: datetime) -> str:
    return f"{_DAYS[value.weekday()]}، {_DATES[value.day]} من {_MONTHS[value.month]} عام {number(value.year)}"


def clock(value: datetime) -> str:
    """Read the supplied local wall clock without rounding seconds or fractions."""
    hour = _HOURS[value.hour % 12]
    minute = value.minute
    if minute == 15:
        hour += " والربع"
    elif minute == 30:
        hour += " والنصف"
    elif minute:
        feminine = ("", "إحدى", "اثنتان", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثمان", "تسع")
        if minute == 1:
            phrase = "دقيقة"
        elif minute == 2:
            phrase = "دقيقتان"
        elif minute <= 10:
            phrase = (feminine[minute] if minute < 10 else "عشر") + " دقائق"
        elif minute < 20:
            phrase = ("إحدى عشرة" if minute == 11 else "اثنتا عشرة" if minute == 12
                      else feminine[minute - 10] + " عشرة") + " دقيقة"
        else:
            tens, ones = divmod(minute, 10)
            phrase = (feminine[ones] + " و" if ones else "") + _TENS[tens] + " دقيقة"
        hour += " و" + phrase
    if value.second or value.microsecond:
        hour += " و" + number(value.second)
        if value.microsecond:
            fraction = str(value.microsecond).zfill(6).rstrip("0")
            hour += " فاصلة " + " ".join(_ONES[int(digit)] for digit in fraction)
        hour += " ثانية"
    period = ("صباحاً" if value.hour < 12 else "ظهراً" if value.hour < 15
              else "عصراً" if value.hour < 18 else "مساءً")
    if value.hour == 0:
        period = "بعد منتصف الليل"
    return f"{hour} {period}"


def slot(value: datetime) -> str:
    return f"يوم {date(value)}، الساعة {clock(value)}"


def zone(name: str) -> str:
    return {"America/Toronto": "بتوقيت تورونتو", "UTC": "بالتوقيت العالمي"}.get(name, f"بتوقيت {name}")


def duration(minutes: int) -> str:
    return "نصف ساعة" if minutes == 30 else "ساعة" if minutes == 60 else f"{number(minutes)} دقيقة"
