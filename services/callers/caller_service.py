"""
=====================================================
AI Voice Platform v2 - Caller Recognition Service
=====================================================
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path
from loguru import logger


class CallerRecognitionService:
    """
    Recognizes callers and remembers their names

    Features:
    - Asks for name on first call
    - Remembers name for returning callers
    - Handles private numbers (doesn't save)
    - Tracks call count and last call
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize caller recognition service

        Args:
            storage_path: Path to store caller data (default: data/caller_names.json)
        """
        if storage_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            storage_path = str(data_dir / "caller_names.json")

        self.storage_path = storage_path
        self._callers: Dict[str, Dict] = {}
        self._load_callers()

    def _load_callers(self):
        """Load caller data from JSON file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self._callers = json.load(f)
                logger.info(f"Loaded {len(self._callers)} caller profiles")
        except Exception as e:
            logger.error(f"Failed to load callers: {e}")
            self._callers = {}

    def _save_callers(self):
        """Save caller data to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._callers, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._callers)} caller profiles")
        except Exception as e:
            logger.error(f"Failed to save callers: {e}")

    def is_known_caller(self, phone_number: str) -> bool:
        """Check if this is a returning caller"""
        # Private/unknown numbers aren't tracked
        if self._is_private_number(phone_number):
            return False
        return phone_number in self._callers

    def get_caller_name(self, phone_number: str) -> Optional[str]:
        """Get caller's name if known"""
        if self._is_private_number(phone_number):
            return None
        caller = self._callers.get(phone_number)
        return caller.get("name") if caller else None

    def get_caller_language(self, phone_number: str) -> Optional[str]:
        """Get caller's preferred language if known"""
        if self._is_private_number(phone_number):
            return None
        caller = self._callers.get(phone_number)
        return caller.get("language") if caller else None

    def get_caller_info(self, phone_number: str) -> Optional[Dict]:
        """Get full caller info"""
        return self._callers.get(phone_number)

    def register_caller(self, phone_number: str, name: str, language: str = "en") -> Dict:
        """
        Register a new caller or update existing

        Args:
            phone_number: Caller's phone number
            name: Caller's name
            language: Detected language

        Returns:
            Updated caller info
        """
        # Don't save private numbers
        if self._is_private_number(phone_number):
            logger.info(f"Private number - not saving: {name}")
            return {
                "name": name,
                "language": language,
                "is_private": True
            }

        now = datetime.now().isoformat()

        if phone_number in self._callers:
            # Update existing caller
            caller = self._callers[phone_number]
            caller["name"] = name
            caller["language"] = language
            caller["last_call"] = now
            caller["call_count"] = caller.get("call_count", 0) + 1
            logger.info(f"Updated returning caller: {name} ({phone_number}) - call #{caller['call_count']}")
        else:
            # New caller
            self._callers[phone_number] = {
                "name": name,
                "language": language,
                "first_call": now,
                "last_call": now,
                "call_count": 1
            }
            logger.info(f"Registered new caller: {name} ({phone_number})")

        self._save_callers()
        return self._callers[phone_number]

    def record_call(self, phone_number: str) -> Optional[Dict]:
        """Record that a caller called (updates stats)"""
        if self._is_private_number(phone_number):
            return None

        if phone_number in self._callers:
            caller = self._callers[phone_number]
            caller["last_call"] = datetime.now().isoformat()
            caller["call_count"] = caller.get("call_count", 0) + 1
            self._save_callers()
            return caller
        return None

    def _is_private_number(self, phone_number: str) -> bool:
        """Check if number is private/unknown"""
        private_indicators = ["private", "unknown", "anonymous", "restricted", ""]
        if not phone_number:
            return True
        return phone_number.lower().strip() in private_indicators

    def get_greeting_for_caller(self, phone_number: str, language: str = "auto") -> str:
        """
        Get personalized greeting for caller

        Args:
            phone_number: Caller's phone number
            language: Current language ('en', 'ar', or 'auto')

        Returns:
            Greeting message
        """
        caller_name = self.get_caller_name(phone_number)
        caller_language = self.get_caller_language(phone_number)

        if caller_name:
            # Returning caller — greet in their preferred language
            lang = caller_language or language
            if lang == "ar":
                return f"هلا {caller_name}! أنا أمل من فليكسبل أكاونتنغ، أهلاً فيك مرة ثانية. كيف أقدر أساعدك اليوم؟"
            else:
                return f"Hey {caller_name}! It's Amal from Flexible Accounting, great to hear from you again. How can I help you today?"
        else:
            # New caller — bilingual greeting, ask for language preference
            return (
                "Hi, thank you for contacting Flexible Accounting! "
                "My name is Amal, and I speak English and Arabic. "
                "Which language do you prefer? "
                "مرحبا، شكراً لاتصالك بفليكسبل أكاونتنغ! "
                "اسمي أمل، وأتكلم عربي وإنجليزي. "
                "أي لغة تفضل؟"
            )

    def get_all_callers(self) -> Dict:
        """Get all caller data"""
        return self._callers.copy()

    def get_recent_callers(self, limit: int = 10) -> list:
        """Get recent callers sorted by last call time"""
        callers = list(self._callers.values())
        callers.sort(key=lambda x: x.get("last_call", ""), reverse=True)
        return callers[:limit]

    def get_frequent_callers(self, limit: int = 5, min_calls: int = 2) -> list:
        """Get most frequent callers (VIPs)"""
        callers = [
            c for c in self._callers.values()
            if c.get("call_count", 0) >= min_calls
        ]
        callers.sort(key=lambda x: x.get("call_count", 0), reverse=True)
        return callers[:limit]


# Global instance
_caller_service: Optional[CallerRecognitionService] = None


def get_caller_service() -> CallerRecognitionService:
    """Get global caller service instance"""
    global _caller_service
    if _caller_service is None:
        _caller_service = CallerRecognitionService()
    return _caller_service
