"""
=====================================================
AI Voice Platform v2 - Accountants Configuration Service
=====================================================
"""

import yaml
import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


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

        self._load_accountants()

    def _load_accountants(self):
        """Load accountants from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Accountants config not found: {self.config_path}")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self._accountants = data.get('accountants', [])
            self._accountants_by_name = {}

            for acc in self._accountants:
                self._accountants_by_name[acc['name'].lower()] = acc

            logger.info(f"Loaded {len(self._accountants)} accountants from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load accountants: {e}")

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

        # Partial match
        for key, acc in self._accountants_by_name.items():
            if name_lower in key or key in name_lower:
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
