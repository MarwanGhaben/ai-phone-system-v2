from datetime import datetime
from zoneinfo import ZoneInfo


BUSINESS_TIMEZONE = ZoneInfo("America/Toronto")


def as_business_time(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=BUSINESS_TIMEZONE)
    return timestamp.astimezone(BUSINESS_TIMEZONE)


def parse_graph_datetime(timestamp: str) -> datetime:
    parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return as_business_time(parsed_timestamp)


def business_now() -> datetime:
    return datetime.now(BUSINESS_TIMEZONE)
