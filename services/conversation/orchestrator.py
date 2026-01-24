"""
=====================================================
AI Voice Platform v2 - Conversation Orchestrator
=====================================================

The orchestrator is the brain of the AI voice platform. It coordinates:
- Twilio Media Streams (bidirectional audio)
- STT service (speech-to-text)
- LLM service (AI reasoning)
- TTS service (text-to-speech)
- Barge-in/interruption handling
- State management
- Knowledge base lookup
"""

import asyncio
import json
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from config.settings import get_settings
from services.stt.deepgram_service import create_deepgram_stt
from services.tts.elevenlabs_service import create_elevenlabs_tts
from services.llm.openai_service import create_openai_llm, Message, LLMRole
from services.telephony.twilio_service import TwilioMediaStreamHandler


class ConversationState(Enum):
    """States of a conversation"""
    CONNECTING = "connecting"
    GREETING = "greeting"
    LISTENING = "listening"  # Waiting for user speech
    THINKING = "thinking"  # AI is processing
    SPEAKING = "speaking"  # AI is talking
    INTERRUPTED = "interrupted"  # User interrupted AI
    CLOSING = "closing"
    ENDED = "ended"


class Intent(Enum):
    """Detected user intents"""
    UNKNOWN = "unknown"
    GREETING = "greeting"
    BOOK_APPOINTMENT = "book_appointment"
    GENERAL_INQUIRY = "general_inquiry"
    TRANSFER_TO_HUMAN = "transfer_to_human"
    PAYROLL_INQUIRY = "payroll_inquiry"
    GOODBYE = "goodbye"


@dataclass
class ConversationContext:
    """Context for a single conversation"""
    call_sid: str
    phone_number: str
    language: str = "auto"  # auto-detect
    state: ConversationState = ConversationState.CONNECTING
    turn_count: int = 0
    detected_intent: Intent = Intent.UNKNOWN
    intent_history: list = field(default_factory=list)
    user_transcript: str = ""
    ai_response: str = ""
    start_time: float = field(default_factory=asyncio.get_event_loop().time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "call_sid": self.call_sid,
            "phone_number": self.phone_number,
            "language": self.language,
            "state": self.state.value,
            "turn_count": self.turn_count,
            "detected_intent": self.detected_intent.value,
            "metadata": self.metadata
        }


class ConversationOrchestrator:
    """
    Main orchestrator for AI voice conversations

    Coordinates all services in real-time:
    1. Receives audio from Twilio
    2. Streams to STT for transcription
    3. Sends transcript to LLM for response
    4. Streams LLM response to TTS for audio
    5. Sends audio back to Twilio
    6. Handles interruption (barge-in) at any time
    """

    def __init__(self):
        """Initialize orchestrator with all services"""
        settings = get_settings()

        # Initialize services
        self.stt = create_deepgram_stt(settings.model_dump())
        self.tts = create_elevenlabs_tts(settings.model_dump())
        self.llm = create_openai_llm(settings.model_dump())

        # Active conversations
        self._conversations: Dict[str, ConversationContext] = {}
        self._twilio_handlers: Dict[str, TwilioMediaStreamHandler] = {}

        # Barge-in detection
        self._energy_threshold: float = settings.interruption_energy_threshold

        # System prompt for the AI
        self._system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are a professional phone receptionist for iFlex Tax, a Canadian accounting firm.

YOUR ROLE:
- Be the first point of contact for callers
- Be warm, professional, and efficient
- Help callers with questions or connect them to the right person

SERVICES OFFERED:
- Personal and corporate tax services
- GST/HST tax return filing
- Payroll services
- Bookkeeping services
- Corporate tax planning

BEHAVIOR:
- Keep responses conversational and brief (2-3 sentences max - this is a phone call)
- Sound natural, like a helpful human
- Don't repeat yourself
- If unsure, offer to transfer to a human

APPOINTMENT BOOKING:
- Accountants available: Hussam Saadaldin, Rami Kahwaji, Abdul
- Ask: Individual or corporate client?
- Ask: Preferred accountant?
- Ask: Preferred date/time?
- Confirm all details before ending

WHEN TO TRANSFER:
- Caller asks to speak with a specific person by name
- Complex tax planning questions
- Client-specific questions (their tax situation, past returns)
- If you don't know the answer
- If caller seems frustrated or confused

TRANSFER SCRIPT:
"Of course, let me transfer you now." (English)
"بالتأكيد، سأحولك الآن." (Arabic)

LANGUAGE DETECTION:
- Automatically detect if caller speaks English or Arabic
- Respond in the same language
- If switching, acknowledge the change

Remember: This is a real phone call. Be concise. Be helpful. Be human."""

    async def handle_call(self, call_sid: str, phone_number: str, websocket) -> None:
        """
        Handle a complete phone call

        Args:
            call_sid: Twilio Call SID
            phone_number: Caller's phone number
            websocket: WebSocket connection from Twilio
        """
        logger.info(f"Orchestrator: Starting call {call_sid} from {phone_number}")

        # Create conversation context
        context = ConversationContext(
            call_sid=call_sid,
            phone_number=phone_number
        )
        self._conversations[call_sid] = context

        # Create Twilio handler
        twilio_handler = TwilioMediaStreamHandler(call_sid, websocket)
        self._twilio_handlers[call_sid] = twilio_handler

        # Set up media handler for incoming audio
        twilio_handler.set_media_handler(self._handle_incoming_audio(call_sid))

        try:
            # Connect to STT
            await self.stt.connect()

            # Handle the WebSocket connection
            await twilio_handler.handle_connection()

        except Exception as e:
            logger.error(f"Orchestrator: Error in call {call_sid}: {e}")
        finally:
            await self.end_call(call_sid)

    async def _handle_incoming_audio(self, call_sid: str):
        """
        Handle incoming audio from caller

        This is called by the Twilio handler when audio arrives.
        It streams the audio to STT and processes the transcript.

        Args:
            call_sid: Call identifier
        """
        async def process_audio(audio_data: bytes):
            """Process audio chunk"""
            context = self._conversations.get(call_sid)
            if not context or context.state == ConversationState.ENDED:
                return

            # Skip if we're speaking and user hasn't interrupted
            if context.state == ConversationState.SPEAKING:
                # Check for barge-in (user speech during AI speech)
                if await self._detect_barge_in(audio_data):
                    logger.info("Orchestrator: User interrupted!")
                    context.state = ConversationState.INTERRUPTED
                    await self.tts.stop()  # Stop TTS
                return

            # Stream to STT
            from services.stt.stt_base import AudioChunk
            await self.stt.stream_audio(AudioChunk(data=audio_data))

        return process_audio

    async def process_transcript(self, call_sid: str, transcript: str, language: str, is_final: bool):
        """
        Process transcript from STT

        Args:
            call_sid: Call identifier
            transcript: Transcribed text
            language: Detected language
            is_final: Whether this is the final transcript
        """
        context = self._conversations.get(call_sid)
        if not context:
            return

        # Update context
        context.user_transcript = transcript
        if context.language == "auto" and language:
            context.language = language

        # Only process final transcripts
        if not is_final:
            return

        logger.info(f"Orchestrator: User said: {transcript}")

        # Detect intent
        intent = await self._detect_intent(transcript, context.language)
        context.detected_intent = intent
        context.intent_history.append(intent.value)

        # Handle based on intent
        if intent == Intent.GOODBYE:
            await self._handle_goodbye(call_sid)
            return

        if intent == Intent.TRANSFER_TO_HUMAN:
            await self._handle_transfer(call_sid)
            return

        # Get AI response
        context.state = ConversationState.THINKING
        response = await self._get_ai_response(call_sid, transcript)

        context.ai_response = response
        context.turn_count += 1
        context.state = ConversationState.SPEAKING

        # Stream response to caller
        await self._speak_to_caller(call_sid, response, context.language)

        # Back to listening
        context.state = ConversationState.LISTENING

    async def _get_ai_response(self, call_sid: str, user_input: str) -> str:
        """
        Get AI response for user input

        Args:
            call_sid: Call identifier
            user_input: User's transcript

        Returns:
            AI response text
        """
        context = self._conversations.get(call_sid)
        if not context:
            return "I apologize, I'm having technical difficulties."

        # Build message list
        messages = [
            Message(role=LLMRole.SYSTEM, content=self._system_prompt)
        ]

        # Add conversation history (last 5 turns)
        # In production, this would come from database
        if context.turn_count > 0:
            # TODO: Fetch from database
            pass

        # Add current user message
        messages.append(Message(role=LLMRole.USER, content=user_input))

        # Get streaming response
        from services.llm.llm_base import LLMRequest, LLMChunk

        request = LLMRequest(
            messages=messages,
            temperature=0.7,
            max_tokens=300,  # Keep responses short for phone
            stream=True
        )

        full_response = ""
        async for chunk in self.llm.chat_stream(request):
            full_response += chunk.delta

        return full_response.strip()

    async def _speak_to_caller(self, call_sid: str, text: str, language: str) -> None:
        """
        Speak response to caller

        Args:
            call_sid: Call identifier
            text: Text to speak
            language: Language code
        """
        logger.info(f"Orchestrator: AI says: {text[:50]}...")

        context = self._conversations.get(call_sid)
        twilio_handler = self._twilio_handlers.get(call_sid)

        if not context or not twilio_handler:
            return

        try:
            # Stream TTS
            from services.tts.tts_base import TTSRequest, TTSChunk

            request = TTSRequest(
                text=text,
                language=language,
                voice_id="Rachel"  # Can be customized per language
            )

            # Note: This sends audio as it's generated
            # For Twilio, we need to convert to μ-law 8kHz
            # This is handled by the TTS service or a converter

            # For now, synthesize full audio then send
            response = await self.tts.synthesize(request)

            # Send to Twilio
            # TODO: Convert MP3 to μ-law 8kHz for Twilio
            await twilio_handler.send_audio(response.audio_data)

        except Exception as e:
            logger.error(f"Orchestrator: Error speaking: {e}")

        finally:
            context.state = ConversationState.LISTENING

    async def _detect_intent(self, text: str, language: str) -> Intent:
        """
        Detect user intent from transcript

        Args:
            text: User's transcript
            language: Language code

        Returns:
            Detected intent
        """
        text_lower = text.lower()

        # Simple keyword matching (can be enhanced with LLM)
        if any(word in text_lower for word in ["goodbye", "bye", "thanks", "thank", "finish"]):
            return Intent.GOODBYE

        if any(word in text_lower for word in ["transfer", "speak to", "talk to", "human", "person"]):
            return Intent.TRANSFER_TO_HUMAN

        if any(word in text_lower for word in ["book", "appointment", "schedule", "reserve"]):
            return Intent.BOOK_APPOINTMENT

        if any(word in text_lower for word in ["payroll", "pay", "salary", "employees"]):
            return Intent.PAYROLL_INQUIRY

        return Intent.GENERAL_INQUIRY

    async def _detect_barge_in(self, audio_data: bytes) -> bool:
        """
        Detect if user is speaking during AI speech (barge-in)

        Args:
            audio_data: Audio chunk to analyze

        Returns:
            True if user is speaking (barge-in detected)
        """
        # Simple energy-based detection
        # Calculate audio energy
        energy = sum(abs(byte - 128) for byte in audio_data[:1000]) / 1000

        if energy > self._energy_threshold * 100:
            return True

        return False

    async def _handle_goodbye(self, call_sid: str) -> None:
        """Handle goodbye intent"""
        logger.info(f"Orchestrator: Call {call_sid} ended (goodbye)")
        await self.end_call(call_sid)

    async def _handle_transfer(self, call_sid: str) -> None:
        """Handle transfer to human intent"""
        logger.info(f"Orchestrator: Call {call_sid} transferring to human")
        # TODO: Implement transfer logic
        await self.end_call(call_sid)

    async def end_call(self, call_sid: str) -> None:
        """
        End a call and cleanup resources

        Args:
            call_sid: Call identifier
        """
        logger.info(f"Orchestrator: Ending call {call_sid}")

        # Update context
        if call_sid in self._conversations:
            self._conversations[call_sid].state = ConversationState.ENDED

        # Disconnect STT
        await self.stt.disconnect()

        # Clean up handlers
        if call_sid in self._twilio_handlers:
            await self._twilio_handlers[call_sid].cleanup()
            del self._twilio_handlers[call_sid]

        # Remove from active conversations
        if call_sid in self._conversations:
            del self._conversations[call_sid]

        # TODO: Save conversation to database

    def get_conversation_context(self, call_sid: str) -> Optional[ConversationContext]:
        """Get conversation context for a call"""
        return self._conversations.get(call_sid)


# Global orchestrator instance
_orchestrator: Optional[ConversationOrchestrator] = None


def get_orchestrator() -> ConversationOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConversationOrchestrator()
    return _orchestrator
