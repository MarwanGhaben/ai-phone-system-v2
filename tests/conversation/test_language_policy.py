import pytest

from services.conversation.language_policy import explicit_language_request


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Arabic", "ar"),
        ("ARABIC PLEASE!", "ar"),
        ("Please speak Arabic.", "ar"),
        ("Can you speak Arabic?", "ar"),
        ("switch to Arabic", "ar"),
        ("arabi", "ar"),
        ("3arabi please", "ar"),
        ("عربي", "ar"),
        ("بالعربي من فضلك", "ar"),
        ("احكي معي بالعربي", "ar"),
        ("تكلم عربي", "ar"),
        ("English", "en"),
        ("english please!", "en"),
        ("Please speak English.", "en"),
        ("Can you speak English?", "en"),
        ("switch to English", "en"),
        ("بالإنجليزي من فضلك", "en"),
        ("تكلم انجليزي", "en"),
        ("My email is naam@example.com", None),
        ("I need an English-speaking accountant", None),
        ("Do not switch to English", None),
        ("I am asking about Arabic documents", None),
        ("Should we use Arabic or English?", None),
        ('She said "please speak Arabic"', None),
        ('"please speak Arabic"', None),
        ("Arabic documents", None),
        ("مرحبا، أريد حجز موعد", None),
        ("", None),
    ],
)
def test_explicit_language_requests_are_whole_utterance_matches(utterance, expected):
    assert explicit_language_request(utterance) == expected
