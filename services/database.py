"""
=====================================================
AI Voice Platform v2 - Async Database Connection Pool
=====================================================
Provides a shared asyncpg connection pool for all services.
"""

import asyncpg
from typing import Optional
from loguru import logger
from config.settings import settings


_pool: Optional[asyncpg.Pool] = None


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
                command_timeout=30,
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    return _pool


async def close_db_pool():
    """Close the connection pool (call on app shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")
