"""
=====================================================
AI Voice Platform v2 - FAQ Knowledge Base Service
=====================================================
"""

import yaml
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from services.knowledge.kb_base import KnowledgeBaseServiceBase, FAQAnswer


@dataclass
class FAQEntry:
    """Single FAQ entry"""
    id: int
    category: str
    question_en: str
    answer_en: str
    question_ar: str
    answer_ar: str
    keywords: List[str] = field(default_factory=list)

    def get_question(self, language: str) -> str:
        return self.question_ar if language == "ar" else self.question_en

    def get_answer(self, language: str) -> str:
        return self.answer_ar if language == "ar" else self.answer_en


class FAQKnowledgeService(KnowledgeBaseServiceBase):
    """
    FAQ-based knowledge service using keyword matching

    Fast, lightweight, and effective for structured FAQ data.
    For more advanced semantic search, consider using the
    SemanticFAQMatcher from the old system.
    """

    def __init__(self, faq_path: str = None):
        """
        Initialize FAQ knowledge service

        Args:
            faq_path: Path to FAQ YAML file
        """
        if faq_path is None:
            # Default path relative to this file
            default_path = Path(__file__).parent.parent.parent / "clients" / "iflextax_faq.yaml"
            faq_path = str(default_path)

        self.faq_path = faq_path
        self._faqs: List[FAQEntry] = []
        self._category_index: Dict[str, List[FAQEntry]] = {}

        self._load_faqs()

    def _load_faqs(self):
        """Load FAQ data from YAML file"""
        try:
            if not os.path.exists(self.faq_path):
                logger.warning(f"FAQ file not found: {self.faq_path}")
                return

            with open(self.faq_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            faq_list = data.get('faqs', [])
            self._faqs = []
            self._category_index = {}

            for item in faq_list:
                faq = FAQEntry(
                    id=item.get('id'),
                    category=item.get('category', ''),
                    question_en=item.get('question_en', ''),
                    answer_en=item.get('answer_en', ''),
                    question_ar=item.get('question_ar', ''),
                    answer_ar=item.get('answer_ar', ''),
                    keywords=item.get('keywords', [])
                )
                self._faqs.append(faq)

                # Build category index
                if faq.category not in self._category_index:
                    self._category_index[faq.category] = []
                self._category_index[faq.category].append(faq)

            logger.info(f"Loaded {len(self._faqs)} FAQ entries from {self.faq_path}")

        except Exception as e:
            logger.error(f"Failed to load FAQs: {e}")

    def _keyword_match_score(self, query: str, faq: FAQEntry, language: str) -> float:
        """
        Calculate keyword matching score

        Args:
            query: User's query
            faq: FAQ entry
            language: Language code

        Returns:
            Score from 0.0 to 1.0
        """
        query_lower = query.lower()
        score = 0.0

        # Get question text for the language
        question = faq.get_question(language)
        answer = faq.get_answer(language)

        # Direct question matching (highest weight)
        query_words = set(query_lower.split())
        question_words = set(question.lower().split())

        # Exact phrase match in question
        if query_lower in question.lower():
            score += 0.9

        # Word overlap in question
        overlap = query_words & question_words
        if overlap:
            score += 0.5 * (len(overlap) / len(query_words))

        # Keyword matching
        for keyword in faq.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in query_lower:
                score += 0.3
            if keyword_lower in answer.lower():
                score += 0.1

        # Category matching (bonus)
        if faq.category.lower() in query_lower:
            score += 0.2

        return min(score, 1.0)

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
        if not query or not query.strip():
            return None

        best_faq = None
        best_score = 0.0

        for faq in self._faqs:
            score = self._keyword_match_score(query, faq, language)
            if score > best_score:
                best_score = score
                best_faq = faq

        if best_faq and best_score >= threshold:
            logger.info(f"FAQ match: '{query}' → '{best_faq.get_question(language)}' (score: {best_score:.3f})")
            return FAQAnswer(
                question=best_faq.get_question(language),
                answer=best_faq.get_answer(language),
                category=best_faq.category,
                confidence=best_score,
                faq_id=best_faq.id,
                language=language
            )
        else:
            logger.info(f"No FAQ match for: '{query}' (best score: {best_score:.3f}, threshold: {threshold})")
            return None

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
        if not query or not query.strip():
            return []

        # Score all FAQs
        scored = []
        for faq in self._faqs:
            score = self._keyword_match_score(query, faq, language)
            if score >= threshold:
                scored.append((faq, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        results = []
        for faq, score in scored[:k]:
            results.append(FAQAnswer(
                question=faq.get_question(language),
                answer=faq.get_answer(language),
                category=faq.category,
                confidence=score,
                faq_id=faq.id,
                language=language
            ))

        logger.info(f"Top {len(results)} FAQ matches for: '{query}'")
        return results

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
        top_matches = await self.find_top_answers(query, language, k=max_faqs, threshold=0.60)

        if not top_matches:
            return ""

        if language == "ar":
            context = "\n\n### قاعدة المعرفة (الأسئلة الشائعة):\n\n"
        else:
            context = "\n\n### Knowledge Base (FAQs):\n\n"

        for i, faq in enumerate(top_matches, 1):
            context += f"**{i}. {faq.question}** ({faq.category})\n"
            context += f"{faq.answer}\n\n"

        return context

    def get_all_categories(self) -> List[str]:
        """Get all FAQ categories"""
        return list(self._category_index.keys())

    def get_all_faqs(self, language: str = "en") -> List[Dict[str, Any]]:
        """Get all FAQs in a language"""
        return [
            {
                "id": faq.id,
                "category": faq.category,
                "question": faq.get_question(language),
                "answer": faq.get_answer(language),
                "keywords": faq.keywords
            }
            for faq in self._faqs
        ]

    def get_faqs_by_category(self, category: str, language: str = "en") -> List[Dict[str, Any]]:
        """Get FAQs by category"""
        if category not in self._category_index:
            return []

        return [
            {
                "id": faq.id,
                "category": faq.category,
                "question": faq.get_question(language),
                "answer": faq.get_answer(language),
                "keywords": faq.keywords
            }
            for faq in self._category_index[category]
        ]


# Global instance
_kb_service: Optional[FAQKnowledgeService] = None


def get_kb_service() -> FAQKnowledgeService:
    """Get global knowledge base service instance"""
    global _kb_service
    if _kb_service is None:
        _kb_service = FAQKnowledgeService()
    return _kb_service


def create_kb_service(faq_path: str = None) -> FAQKnowledgeService:
    """
    Factory function to create knowledge base service

    Args:
        faq_path: Optional path to FAQ YAML file

    Returns:
        FAQKnowledgeService instance
    """
    return FAQKnowledgeService(faq_path=faq_path)
