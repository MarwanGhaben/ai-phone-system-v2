"""
=====================================================
AI Voice Platform v2 - Knowledge Base Service Interface
=====================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class FAQAnswer:
    """FAQ answer result"""
    question: str
    answer: str
    category: str
    confidence: float
    faq_id: int
    language: str


class KnowledgeBaseServiceBase(ABC):
    """
    Abstract base for knowledge base services

    Provides FAQ and knowledge lookup capabilities.
    """

    @abstractmethod
    async def find_answer(self, query: str, language: str = "en",
                         threshold: float = 0.75) -> Optional[FAQAnswer]:
        """
        Find the best answer for a query

        Args:
            query: User's question
            language: Language code ('en' or 'ar')
            threshold: Minimum confidence threshold

        Returns:
            FAQAnswer if found above threshold, None otherwise
        """
        pass

    @abstractmethod
    async def find_top_answers(self, query: str, language: str = "en",
                              k: int = 5, threshold: float = 0.75) -> List[FAQAnswer]:
        """
        Find top K answers for a query

        Args:
            query: User's question
            language: Language code ('en' or 'ar')
            k: Number of answers to return
            threshold: Minimum confidence threshold

        Returns:
            List of FAQAnswer sorted by confidence
        """
        pass

    @abstractmethod
    async def get_context_for_prompt(self, query: str, language: str = "en",
                                    max_faqs: int = 5) -> str:
        """
        Get formatted FAQ context for injection into AI prompt

        Args:
            query: User's question
            language: Language code ('en' or 'ar')
            max_faqs: Maximum number of FAQs to include

        Returns:
            Formatted context string
        """
        pass

    @abstractmethod
    def get_all_categories(self) -> List[str]:
        """Get all FAQ categories"""
        pass

    @abstractmethod
    def get_all_faqs(self, language: str = "en") -> List[Dict[str, Any]]:
        """Get all FAQs in a language"""
        pass
