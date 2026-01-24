"""
=====================================================
AI Voice Platform v2 - STT Service Base Interface
=====================================================
Abstract base class for Speech-to-Text providers
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio


class STTStatus(Enum):
    """STT stream status"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class STTResult:
    """Result from STT processing"""
    text: str
    language: str
    confidence: float
    is_final: bool = False
    alternatives: list = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AudioChunk:
    """Audio chunk for streaming"""
    data: bytes
    sample_rate: int = 8000
    channels: int = 1
    is_final: bool = False  # Last chunk of the stream


class STTServiceBase(ABC):
    """
    Abstract base class for Speech-to-Text services

    All STT providers (Deepgram, AssemblyAI, Google, etc.)
    must implement this interface for swapability.
    """

    def __init__(self, api_key: str, language: str = "en-US"):
        """
        Initialize STT service

        Args:
            api_key: Provider API key
            language: Default language code
        """
        self.api_key = api_key
        self.language = language
        self._status = STTStatus.DISCONNECTED
        self._callbacks = []

    @property
    def status(self) -> STTStatus:
        """Get current connection status"""
        return self._status

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to STT service

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to STT service"""
        pass

    @abstractmethod
    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to STT service

        Args:
            audio_chunk: Audio data to transcribe
        """
        pass

    @abstractmethod
    async def get_transcript(self) -> AsyncIterator[STTResult]:
        """
        Get transcription results as they arrive

        Yields:
            STTResult objects with transcribed text
        """
        pass

    @abstractmethod
    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect language from audio

        Args:
            audio_data: Audio sample for detection

        Returns:
            Detected language code (e.g., "en", "ar")
        """
        pass

    def register_transcript_callback(self, callback):
        """
        Register a callback for transcription results

        Args:
            callback: Async function called with STTResult
        """
        self._callbacks.append(callback)

    async def _emit_transcript(self, result: STTResult) -> None:
        """Emit transcript to all registered callbacks"""
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)

    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.disconnect()
