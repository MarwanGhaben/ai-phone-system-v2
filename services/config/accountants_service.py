"""
=====================================================
AI Voice Platform v2 - Accountants Configuration Service
=====================================================
"""

import yaml
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


@dataclass(frozen=True)
class BookingSelection:
    name: str
    staff_id: str
    service_id: str


@dataclass(frozen=True)
class BookingSelectionResult:
    status: str
    selection: BookingSelection | None = None


def _booking_key(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


def _booking_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _build_booking_index(data: object) -> dict[str, tuple[BookingSelection, ...]] | None:
    if not isinstance(data, dict) or not isinstance(data.get("accountants"), list):
        return None
    rows = data["accountants"]
    if not rows:
        return None
    index: dict[str, set[BookingSelection]] = {}
    staff_ids: set[str] = set()
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        name, name_ar = row.get("name"), row.get("name_ar")
        staff_id, service_id = row.get("staff_id"), row.get("service_id")
        aliases, booking_aliases = row.get("aliases", []), row.get("booking_aliases")
        if (not isinstance(name, str) or not _booking_key(name)
                or not isinstance(name_ar, str) or not _booking_key(name_ar)
                or not _booking_id(staff_id) or not _booking_id(service_id)
                or not isinstance(aliases, list)
                or not all(isinstance(alias, str) and _booking_key(alias)
                           for alias in aliases)
                or not isinstance(booking_aliases, list)
                or not all(isinstance(alias, str) and _booking_key(alias)
                           for alias in booking_aliases)):
            return None
        canonical = _booking_key(name)
        if staff_id in staff_ids or canonical in names:
            return None
        staff_ids.add(staff_id)
        names.add(canonical)
        selection = BookingSelection(name, staff_id, service_id)
        for value in (name, name_ar, *booking_aliases):
            index.setdefault(_booking_key(value), set()).add(selection)
    return {key: tuple(values) for key, values in index.items()}


class AccountantsService:
    """
    Loads and provides accountant configuration from YAML file.

    This allows updating accountant names without code changes or rebuilds.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize accountants service

        Args:
            config_path: Path to accountants.yaml file
        """
        if config_path is None:
            # Default path relative to this file
            default_path = Path(__file__).parent.parent.parent / "clients" / "accountants.yaml"
            config_path = str(default_path)

        self.config_path = config_path
        self._accountants: List[Dict[str, Any]] = []
        self._accountants_by_name: Dict[str, Dict[str, Any]] = {}
        self._booking_index: dict[str, tuple[BookingSelection, ...]] | None = None

        self._load_accountants()

    def _load_accountants(self):
        """Load accountants from YAML file"""
        self._booking_index = None
        try:
            if not os.path.exists(self.config_path):
                logger.warning("Accountants config not found")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self._booking_index = _build_booking_index(data)
            self._accountants = data.get('accountants', [])
            self._accountants_by_name = {}

            for acc in self._accountants:
                # Add main name (English)
                self._accountants_by_name[acc['name'].lower()] = acc

                # Add Arabic name if present
                if acc.get('name_ar'):
                    self._accountants_by_name[acc['name_ar']] = acc

                # Add aliases if present (for name variations like "Hossam" vs "Hussam")
                for alias in acc.get('aliases', []):
                    self._accountants_by_name[alias.lower()] = acc

            logger.info("Accountants config loaded")

        except Exception:
            self._booking_index = None
            logger.error("Failed to load accountants config")

    def resolve_booking_accountant(self, name: object) -> BookingSelectionResult:
        """Resolve only an explicit booking name against validated configured IDs."""
        if self._booking_index is None:
            return BookingSelectionResult("invalid_configuration")
        if not isinstance(name, str) or not _booking_key(name):
            return BookingSelectionResult("not_found")
        matches = self._booking_index.get(_booking_key(name), ())
        if len(matches) > 1:
            return BookingSelectionResult("ambiguous")
        if not matches:
            return BookingSelectionResult("not_found")
        return BookingSelectionResult("resolved", matches[0])

    def get_all_accountants(self) -> List[Dict[str, Any]]:
        """Get all accountants"""
        return self._accountants

    def get_accountant_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get accountant by name (fuzzy matching)

        Args:
            name: Accountant name (partial match works)

        Returns:
            Accountant dict or None
        """
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in self._accountants_by_name:
            return self._accountants_by_name[name_lower]

        # Partial match (substring)
        for key, acc in self._accountants_by_name.items():
            if name_lower in key or key in name_lower:
                return acc

        # Fuzzy match — first name starts with same prefix (at least 3 chars)
        if len(name_lower) >= 3:
            for key, acc in self._accountants_by_name.items():
                # Compare first 3+ characters for close matches
                key_first = key.split()[0] if ' ' in key else key
                if (key_first.startswith(name_lower[:3]) or
                    name_lower.startswith(key_first[:3])):
                    logger.info(f"Fuzzy matched '{name}' to '{acc['name']}'")
                    return acc

        return None

    def get_names(self, language: str = "en") -> List[str]:
        """
        Get list of accountant names

        Args:
            language: 'en' or 'ar'

        Returns:
            List of names
        """
        field = 'name_ar' if language == 'ar' else 'name'
        return [acc.get(field, acc['name']) for acc in self._accountants]

    def get_names_formatted(self, language: str = "en") -> str:
        """
        Get accountant names as a formatted string for AI prompt

        Args:
            language: 'en' or 'ar'

        Returns:
            Formatted string like "Name1, Name2, and Name3"
        """
        names = self.get_names(language)
        if len(names) == 0:
            return ""
        elif len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + ", and " + names[-1]

    def reload(self):
        """Reload accountants from file (use after editing YAML)"""
        self._load_accountants()


# Global instance
_accountants_service: AccountantsService = None


def get_accountants_service() -> AccountantsService:
    """Get global accountants service instance"""
    global _accountants_service
    if _accountants_service is None:
        _accountants_service = AccountantsService()
    return _accountants_service
