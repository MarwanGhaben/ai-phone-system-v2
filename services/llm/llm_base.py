"""
=====================================================
AI Voice Platform v2 - LLM Service Base Interface
=====================================================
Abstract base class for LLM (Large Language Model) providers
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class LLMRole(Enum):
    """Roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """Conversation message"""
    role: LLMRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API calls"""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class LLMRequest:
    """Request for LLM completion"""
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = True
    tools: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    role: LLMRole = LLMRole.ASSISTANT
    finish_reason: Optional[str] = None
    tokens_used: int = 0
    tool_calls: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMChunk:
    """Chunk of streaming LLM response"""
    delta: str  # New text since last chunk
    content: str  # Complete content so far
    is_final: bool = False
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMServiceBase(ABC):
    """
    Abstract base class for LLM services

    All LLM providers (OpenAI, Anthropic, etc.) must implement this interface.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize LLM service

        Args:
            api_key: Provider API key
            model: Model identifier
        """
        self.api_key = api_key
        self.model = model
        self._conversation_history: List[Message] = []

    @abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        Non-streaming chat completion

        Args:
            request: LLM request

        Returns:
            Complete LLM response
        """
        pass

    @abstractmethod
    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """
        Streaming chat completion

        Args:
            request: LLM request

        Yields:
            LLMChunk objects as tokens are generated
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (for cost tracking)

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        pass

    def add_to_history(self, message: Message) -> None:
        """Add message to conversation history"""
        self._conversation_history.append(message)

    def clear_history(self) -> None:
        """Clear conversation history"""
        self._conversation_history = []

    def get_history(self) -> List[Message]:
        """Get conversation history"""
        return self._conversation_history.copy()

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.clear_history()
