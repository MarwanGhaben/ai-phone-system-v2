from typing import Optional

import asyncpg


_CALL_SUMMARY_QUERY = """
WITH caller_counts AS (
    SELECT phone_number, COUNT(*) AS call_count
    FROM call_logs
    WHERE phone_number IS NOT NULL
    GROUP BY phone_number
)
SELECT
    COUNT(*)::INTEGER AS total_calls,
    COUNT(DISTINCT phone_number)::INTEGER AS unique_callers,
    (SELECT COUNT(*) FROM caller_counts WHERE call_count > 1)::INTEGER
        AS returning_callers,
    ROUND(AVG(duration_seconds)::NUMERIC, 1) AS avg_duration_seconds,
    COUNT(duration_seconds)::INTEGER AS duration_sample_count,
    CASE WHEN COUNT(*) FILTER (WHERE ended_at IS NOT NULL) > 0 THEN
        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE ended_at IS NOT NULL AND transfer_requested IS TRUE
            ) / COUNT(*) FILTER (WHERE ended_at IS NOT NULL),
            1
        )
    END AS transfer_rate,
    COUNT(*) FILTER (WHERE ended_at IS NOT NULL)::INTEGER
        AS transfer_sample_count
FROM call_logs
"""

_LANGUAGE_DISTRIBUTION_QUERY = """
SELECT COALESCE(language, 'en') AS language, COUNT(*)::INTEGER AS count
FROM call_logs
GROUP BY COALESCE(language, 'en')
ORDER BY count DESC
"""


async def fetch_call_statistics(pool: asyncpg.Pool) -> dict:
    summary = await pool.fetchrow(_CALL_SUMMARY_QUERY)
    languages = await pool.fetch(_LANGUAGE_DISTRIBUTION_QUERY)
    total_calls = int(summary["total_calls"] or 0)
    return {
        "total_calls": total_calls,
        "unique_callers": int(summary["unique_callers"] or 0),
        "returning_callers": int(summary["returning_callers"] or 0),
        "avg_duration_seconds": _optional_float(summary["avg_duration_seconds"]),
        "duration_sample_count": int(summary["duration_sample_count"] or 0),
        "transfer_rate": _optional_float(summary["transfer_rate"]),
        "transfer_sample_count": int(summary["transfer_sample_count"] or 0),
        "language_distribution": _language_distribution(languages, total_calls),
    }


def _optional_float(measurement) -> Optional[float]:
    return float(measurement) if measurement is not None else None


def _language_distribution(languages, total_calls: int) -> list[dict]:
    if not total_calls:
        return []
    return [
        {
            "language": "Arabic" if row["language"] == "ar" else "English",
            "code": row["language"],
            "count": int(row["count"]),
            "percentage": round(row["count"] / total_calls * 100, 1),
        }
        for row in languages
    ]
