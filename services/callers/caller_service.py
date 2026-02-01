"""
=====================================================
AI Voice Platform v2 - Caller Recognition Service
=====================================================
PostgreSQL-backed caller recognition and profiles.
"""

from datetime import datetime
from typing import Optional, Dict, List
from loguru import logger

from services.database import get_db_pool


class CallerRecognitionService:
    """
    Recognizes callers and remembers their names.

    Features:
    - Asks for name on first call
    - Remembers name for returning callers
    - Handles private numbers (doesn't save)
    - Tracks call count and last call
    """

    # ---- helpers ------------------------------------------------

    @staticmethod
    def _is_private_number(phone_number: str) -> bool:
        """Check if number is private/unknown"""
        private_indicators = ["private", "unknown", "anonymous", "restricted", ""]
        if not phone_number:
            return True
        return phone_number.lower().strip() in private_indicators

    @staticmethod
    def _row_to_dict(row) -> Dict:
        """Convert an asyncpg Record to the legacy dict format"""
        if row is None:
            return None
        return {
            "name": row["name"],
            "language": row["language"],
            "call_count": row["call_count"],
            "first_call": row["first_call"].isoformat() if row["first_call"] else None,
            "last_call": row["last_call"].isoformat() if row["last_call"] else None,
        }

    # ---- public API (async) -------------------------------------

    async def is_known_caller(self, phone_number: str) -> bool:
        """Check if this is a returning caller"""
        if self._is_private_number(phone_number):
            return False
        pool = await get_db_pool()
        row = await pool.fetchval(
            "SELECT 1 FROM callers WHERE phone_number = $1", phone_number
        )
        return row is not None

    async def get_caller_name(self, phone_number: str) -> Optional[str]:
        """Get caller's name if known"""
        if self._is_private_number(phone_number):
            return None
        pool = await get_db_pool()
        return await pool.fetchval(
            "SELECT name FROM callers WHERE phone_number = $1", phone_number
        )

    async def get_caller_language(self, phone_number: str) -> Optional[str]:
        """Get caller's preferred language if known"""
        if self._is_private_number(phone_number):
            return None
        pool = await get_db_pool()
        return await pool.fetchval(
            "SELECT language FROM callers WHERE phone_number = $1", phone_number
        )

    async def get_caller_info(self, phone_number: str) -> Optional[Dict]:
        """Get full caller info"""
        if self._is_private_number(phone_number):
            return None
        pool = await get_db_pool()
        row = await pool.fetchrow(
            "SELECT name, language, call_count, first_call, last_call "
            "FROM callers WHERE phone_number = $1",
            phone_number,
        )
        return self._row_to_dict(row)

    async def register_caller(
        self, phone_number: str, name: str, language: str = "en"
    ) -> Dict:
        """
        Register a new caller or update existing.

        Returns:
            Updated caller info dict
        """
        if self._is_private_number(phone_number):
            logger.info(f"Private number - not saving: {name}")
            return {"name": name, "language": language, "is_private": True}

        pool = await get_db_pool()
        now = datetime.now()

        # Upsert: insert or update on conflict
        row = await pool.fetchrow(
            """
            INSERT INTO callers (phone_number, name, language, call_count, first_call, last_call)
            VALUES ($1, $2, $3, 1, $4, $4)
            ON CONFLICT (phone_number) DO UPDATE
                SET name = EXCLUDED.name,
                    language = EXCLUDED.language,
                    last_call = EXCLUDED.last_call,
                    call_count = callers.call_count + 1
            RETURNING name, language, call_count, first_call, last_call
            """,
            phone_number, name, language, now,
        )
        result = self._row_to_dict(row)
        logger.info(
            f"Registered/updated caller: {name} ({phone_number}) "
            f"- call #{result['call_count']}"
        )
        return result

    async def record_call(self, phone_number: str) -> Optional[Dict]:
        """Record that a caller called (increment stats)"""
        if self._is_private_number(phone_number):
            return None

        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            UPDATE callers
            SET last_call = NOW(), call_count = call_count + 1
            WHERE phone_number = $1
            RETURNING name, language, call_count, first_call, last_call
            """,
            phone_number,
        )
        return self._row_to_dict(row)

    def get_greeting_for_caller(self, caller_name: Optional[str], caller_language: Optional[str] = None) -> str:
        """
        Get personalized greeting for caller.

        Args:
            caller_name: Caller's name (None if unknown)
            caller_language: Caller's preferred language

        Returns:
            Greeting message
        """
        if caller_name:
            lang = caller_language or "en"
            if lang == "ar":
                return (
                    f"هلا {caller_name}! أنا سارة من فليكسبل أكاونتنغ، "
                    "أهلاً فيك مرة ثانية. كيف أقدر أساعدك اليوم؟"
                )
            else:
                return (
                    f"Hey {caller_name}! It's Sarah from Flexible Accounting, "
                    "great to hear from you again. How can I help you today?"
                )
        else:
            return (
                "Hi, thank you for contacting Flexible Accounting! "
                "My name is Sarah, and I speak English and Arabic. "
                "Which language do you prefer?"
            )

    async def get_all_callers(self) -> Dict[str, Dict]:
        """Get all caller data keyed by phone number"""
        pool = await get_db_pool()
        rows = await pool.fetch(
            "SELECT phone_number, name, language, call_count, first_call, last_call "
            "FROM callers ORDER BY last_call DESC"
        )
        return {row["phone_number"]: self._row_to_dict(row) for row in rows}

    async def get_recent_callers(self, limit: int = 10) -> List[Dict]:
        """Get recent callers sorted by last call time"""
        pool = await get_db_pool()
        rows = await pool.fetch(
            "SELECT name, language, call_count, first_call, last_call "
            "FROM callers ORDER BY last_call DESC LIMIT $1",
            limit,
        )
        return [self._row_to_dict(row) for row in rows]

    async def get_frequent_callers(self, limit: int = 5, min_calls: int = 2) -> List[Dict]:
        """Get most frequent callers (VIPs)"""
        pool = await get_db_pool()
        rows = await pool.fetch(
            "SELECT name, language, call_count, first_call, last_call "
            "FROM callers WHERE call_count >= $1 "
            "ORDER BY call_count DESC LIMIT $2",
            min_calls, limit,
        )
        return [self._row_to_dict(row) for row in rows]


# Global instance
_caller_service: Optional[CallerRecognitionService] = None


def get_caller_service() -> CallerRecognitionService:
    """Get global caller service instance"""
    global _caller_service
    if _caller_service is None:
        _caller_service = CallerRecognitionService()
    return _caller_service
