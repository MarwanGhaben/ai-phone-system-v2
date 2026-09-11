"""
=====================================================
AI Voice Platform v2 - Async Database Connection Pool
=====================================================
Provides a shared asyncpg connection pool for all services.
"""

import asyncio
import asyncpg
from typing import Optional
from loguru import logger
from config.settings import settings
from migrations.schema_contract import check_runtime_compatibility


_pool: Optional[asyncpg.Pool] = None
POOL_CONNECT_TIMEOUT = 10
READINESS_ACQUIRE_TIMEOUT = 5
READINESS_QUERY_TIMEOUT = 5


class DatabaseReadinessError(Exception):
    """Safe application-facing database readiness failure."""


async def get_db_pool() -> asyncpg.Pool:
    """
    Get or create the shared asyncpg connection pool.

    Returns:
        asyncpg.Pool: The connection pool
    """
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=10,
                timeout=POOL_CONNECT_TIMEOUT,
                command_timeout=30,
            )
            logger.info("Database connection pool created successfully")
        except Exception:
            logger.error("Failed to create database pool")
            raise DatabaseReadinessError("database unavailable") from None
    return _pool


async def check_database_compatibility(pool: Optional[asyncpg.Pool] = None) -> None:
    """Run the bounded, read-only structural and migration-history check."""
    active_pool = pool or _pool
    if active_pool is None:
        raise DatabaseReadinessError("database unavailable")
    try:
        async with asyncio.timeout(READINESS_ACQUIRE_TIMEOUT):
            async with active_pool.acquire() as conn:
                async with asyncio.timeout(READINESS_QUERY_TIMEOUT):
                    async with conn.transaction(
                        isolation="repeatable_read", readonly=True
                    ):
                        await conn.execute(
                            "SET LOCAL statement_timeout = '4000ms'; "
                            "SET LOCAL lock_timeout = '3000ms'; "
                            "SET LOCAL search_path = pg_catalog")
                        await check_runtime_compatibility(conn)
    except Exception:
        raise DatabaseReadinessError("database unavailable") from None


async def close_db_pool():
    """Close the connection pool (call on app shutdown)."""
    global _pool
    pool, _pool = _pool, None
    if pool is None:
        return
    try:
        async with asyncio.timeout(5):
            await pool.close()
    except asyncio.CancelledError:
        pool.terminate()
        raise
    except Exception:
        pool.terminate()
    logger.info("Database connection pool closed")
