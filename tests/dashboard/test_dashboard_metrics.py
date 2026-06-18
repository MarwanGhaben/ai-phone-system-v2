from decimal import Decimal
from pathlib import Path

import pytest

from services.dashboard.dashboard_routes import get_pipeline_latency
from services.dashboard.dashboard_service import DashboardService
from services.dashboard.metrics_service import fetch_call_statistics


class CallMetricsPool:
    def __init__(self, summary: dict, languages: list[dict]) -> None:
        self.summary = summary
        self.languages = languages

    async def fetchrow(self, query: str):
        return self.summary

    async def fetch(self, query: str):
        return self.languages


@pytest.mark.asyncio
async def test_call_statistics_use_persisted_call_logs() -> None:
    pool = CallMetricsPool(
        {
            "total_calls": 4,
            "unique_callers": 3,
            "returning_callers": 1,
            "avg_duration_seconds": Decimal("82.5"),
            "duration_sample_count": 4,
            "transfer_rate": Decimal("25.0"),
            "transfer_sample_count": 3,
        },
        [
            {"language": "en", "count": 3},
            {"language": "ar", "count": 1},
        ],
    )

    statistics = await fetch_call_statistics(pool)

    assert statistics["total_calls"] == 4
    assert statistics["unique_callers"] == 3
    assert statistics["returning_callers"] == 1
    assert statistics["avg_duration_seconds"] == 82.5
    assert statistics["duration_sample_count"] == 4
    assert statistics["transfer_rate"] == 25.0
    assert statistics["transfer_sample_count"] == 3
    assert statistics["language_distribution"][0]["percentage"] == 75.0


@pytest.mark.asyncio
async def test_empty_call_logs_have_zero_counts_and_unavailable_samples() -> None:
    pool = CallMetricsPool(
        {
            "total_calls": 0,
            "unique_callers": 0,
            "returning_callers": 0,
            "avg_duration_seconds": None,
            "duration_sample_count": 0,
            "transfer_rate": None,
            "transfer_sample_count": 0,
        },
        [],
    )

    statistics = await fetch_call_statistics(pool)

    assert statistics["total_calls"] == 0
    assert statistics["avg_duration_seconds"] is None
    assert statistics["duration_sample_count"] == 0
    assert statistics["transfer_rate"] is None
    assert statistics["language_distribution"] == []


@pytest.mark.asyncio
async def test_pipeline_latency_is_explicitly_unavailable() -> None:
    latency = await get_pipeline_latency(user={})

    assert latency["available"] is False
    assert latency["total_ms"] is None
    assert latency["samples"] == 0


@pytest.mark.asyncio
async def test_legacy_uninstrumented_metrics_are_not_numeric_zeroes() -> None:
    dashboard = DashboardService()

    api_monitoring = await dashboard.get_api_monitoring()
    bookings = await dashboard.get_bookings_overview()
    error_rate = dashboard._get_error_rate()

    assert api_monitoring["available"] is False
    assert all(service["spent"] is None for service in api_monitoring["services"])
    assert bookings["available"] is False
    assert bookings["stats"]["total"] is None
    assert error_rate["available"] is False
    assert error_rate["current"] is None


def test_dashboard_labels_missing_average_duration_honestly() -> None:
    dashboard_html = Path("templates/dashboard.html").read_text(encoding="utf-8")

    assert "data.avg_duration_seconds === null" in dashboard_html
    assert "Not available" in dashboard_html
