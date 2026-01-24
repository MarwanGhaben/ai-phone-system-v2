"""
=====================================================
AI Voice Platform v2 - TTS Service Base Interface
=====================================================
Abstract base class for Text-to-Speech providers
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio


class TTSStatus(Enum):
    """TTS service status"""
    IDLE = "idle"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class TTSRequest:
    """Request for TTS synthesis"""
    text: str
    language: str = "en"
    voice_id: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TTSResponse:
    """Response from TTS synthesis"""
    audio_data: bytes
    sample_rate: int
    format: str  # mp3, pcm, ulaw, etc.
    duration_ms: int
    text: str
    is_final: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TTSChunk:
    """Chunk of streaming TTS audio"""
    audio_data: bytes
    is_final: bool = False
    text_offset: int = 0  # Character position in original text


class TTSServiceBase(ABC):
    """
    Abstract base class for Text-to-Speech services

    All TTS providers (ElevenLabs, Google, Azure, etc.)
    must implement this interface for swapability.
    """

    def __init__(self, api_key: str, default_voice_id: str):
        """
        Initialize TTS service

        Args:
            api_key: Provider API key
            default_voice_id: Default voice to use
        """
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        self._status = TTSStatus.IDLE
        self._current_request: Optional[TTSRequest] = None

    @property
    def status(self) -> TTSStatus:
        """Get current synthesis status"""
        return self._status

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._status == TTSStatus.SPEAKING

    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text (blocking)

        Args:
            request: TTS request with text and options

        Returns:
            TTS response with audio data
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[TTSChunk]:
        """
        Synthesize speech with streaming output

        Args:
            request: TTS request with text and options

        Yields:
            TTSChunk objects as audio is generated
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop current synthesis (for interruption)"""
        pass

    @abstractmethod
    async def get_available_voices(self, language: str = "all") -> list:
        """
        Get list of available voices

        Args:
            language: Filter by language ('all' for no filter)

        Returns:
            List of voice metadata
        """
        pass

    async def interrupt(self) -> None:
        """Interrupt current speech (for barge-in)"""
        if self._status == TTSStatus.SPEAKING:
            self._status = TTSStatus.INTERRUPTED
            await self.stop()

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.stop()
