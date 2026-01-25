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

    # Caller recognition
    caller_name: Optional[str] = None
    is_known_caller: bool = False
    name_collected: bool = False  # True if we already got their name

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "call_sid": self.call_sid,
            "phone_number": self.phone_number,
            "language": self.language,
            "state": self.state.value,
            "turn_count": self.turn_count,
            "detected_intent": self.detected_intent.value,
            "caller_name": self.caller_name,
            "is_known_caller": self.is_known_caller,
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

        # Initialize services (may return None if not available)
        self.stt = create_deepgram_stt(settings.model_dump())
        self.tts = create_elevenlabs_tts(settings.model_dump())
        self.llm = create_openai_llm(settings.model_dump())

        # Log service availability
        if not self.tts:
            logger.warning("TTS service not available - audio responses will be limited")

        # Active conversations
        self._conversations: Dict[str, ConversationContext] = {}
        self._twilio_handlers: Dict[str, TwilioMediaStreamHandler] = {}

        # Barge-in detection
        self._energy_threshold: float = settings.interruption_energy_threshold

        # System prompt for the AI
        self._system_prompt = self._get_system_prompt()

        # Caller recognition service
        from services.callers.caller_service import get_caller_service
        self.caller_service = get_caller_service()

        # Track which calls have greeted
        self._greeted_calls: set = set()

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        # Get accountant names from service
        from services.config.accountants_service import get_accountants_service
        acc_service = get_accountants_service()
        accountant_names_en = acc_service.get_names_formatted("en")
        accountant_names_ar = acc_service.get_names_formatted("ar")

        return f"""You are a professional phone receptionist for iFlex Tax, a Canadian accounting firm.

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
- Accountants available: {accountant_names_en}
- In Arabic: {accountant_names_ar}
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

    async def handle_call(self, call_sid: str, phone_number: str, websocket, stream_sid: str = "") -> None:
        """
        Handle a complete phone call

        Args:
            call_sid: Twilio Call SID
            phone_number: Caller's phone number
            websocket: WebSocket connection from Twilio
            stream_sid: Twilio Media Stream SID (for sending audio back)
        """
        import traceback
        logger.info(f"Orchestrator: ===== START CALL {call_sid} from {phone_number}, stream_sid={stream_sid}")

        try:
            # Check if this is a returning caller
            logger.info(f"Orchestrator: Getting caller info for {phone_number}")
            caller_info = self.caller_service.get_caller_info(phone_number)
            caller_name = caller_info.get("name") if caller_info else None
            caller_language = caller_info.get("language", "auto") if caller_info else "auto"

            logger.info(f"Orchestrator: Caller info - name={caller_name}, language={caller_language}")

            # Create conversation context
            context = ConversationContext(
                call_sid=call_sid,
                phone_number=phone_number,
                language=caller_language
            )
            context.caller_name = caller_name
            context.is_known_caller = caller_name is not None
            self._conversations[call_sid] = context
            logger.info(f"Orchestrator: Conversation context created")

            # Create Twilio handler
            logger.info(f"Orchestrator: Creating Twilio handler")
            twilio_handler = TwilioMediaStreamHandler(call_sid, stream_sid, websocket)
            self._twilio_handlers[call_sid] = twilio_handler

            # IMPORTANT: Manually set streaming flag since we already consumed the start event in main.py
            # The handler's handle_connection() won't see the start event, so we set this manually
            twilio_handler._is_streaming = True
            twilio_handler._is_connected = True
            logger.info(f"Orchestrator: Handler configured - streaming=True, stream_sid={stream_sid}")

            # Set up media handler for incoming audio
            twilio_handler.set_media_handler(self._handle_incoming_audio(call_sid))
            logger.info(f"Orchestrator: Media handler registered")

            # Connect to STT
            logger.info(f"Orchestrator: Connecting to STT...")
            await self.stt.connect()
            logger.info(f"Orchestrator: STT connected")

            # Get personalized greeting and send it immediately
            logger.info(f"Orchestrator: Getting greeting...")
            greeting = self.caller_service.get_greeting_for_caller(phone_number, caller_language)
            logger.info(f"Orchestrator: Greeting: '{greeting[:50]}...'")

            # Send greeting immediately (before starting event loop)
            # This ensures state is properly managed and no race conditions
            logger.info(f"Orchestrator: Sending greeting...")
            context.state = ConversationState.SPEAKING
            try:
                await self._speak_to_caller(call_sid, greeting, caller_language)
                logger.info(f"Orchestrator: Greeting sent successfully")
            except Exception as e:
                logger.error(f"Orchestrator: Failed to send greeting: {e}")
            finally:
                context.state = ConversationState.LISTENING

            # Small delay after greeting before starting event loop
            await asyncio.sleep(0.5)

            # Start transcript consumer as background task
            # This consumes transcripts from STT queue and processes them
            logger.info(f"Orchestrator: Starting transcript consumer...")
            transcript_task = asyncio.create_task(
                self._consume_stt_transcripts(call_sid),
                name=f"transcript_consumer_{call_sid}"
            )

            # Now handle the WebSocket connection (incoming audio)
            # This runs concurrently with transcript consumer
            logger.info(f"Orchestrator: Starting connection handler...")
            try:
                await twilio_handler.handle_connection()
            finally:
                # Cancel transcript task when connection ends
                if not transcript_task.done():
                    logger.info(f"Orchestrator: Cancelling transcript consumer...")
                    transcript_task.cancel()
                    try:
                        await transcript_task
                    except asyncio.CancelledError:
                        pass
            logger.info(f"Orchestrator: WebSocket handling complete")

        except Exception as e:
            logger.error(f"Orchestrator: Exception in call {call_sid}: {e}")
            logger.error(f"Orchestrator: Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"Orchestrator: ===== END CALL {call_sid}")
            await self.end_call(call_sid)

    def _handle_incoming_audio(self, call_sid: str):
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
                    if self.tts:
                        await self.tts.stop()  # Stop TTS
                return

            # Stream to STT
            from services.stt.stt_base import AudioChunk
            await self.stt.stream_audio(AudioChunk(data=audio_data))

        return process_audio

    async def _consume_stt_transcripts(self, call_sid: str) -> None:
        """
        Consume transcripts from STT service queue and process them

        This runs as a background task alongside the WebSocket event loop.
        It continuously polls for transcripts and processes them.

        Args:
            call_sid: Call identifier
        """
        context = self._conversations.get(call_sid)
        if not context:
            logger.warning(f"Orchestrator: No context found for call {call_sid}")
            return

        logger.info(f"Orchestrator: Transcript consumer started for call {call_sid}")

        try:
            # get_transcript() returns an async iterator
            async for result in self.stt.get_transcript():
                if not result or not result.text:
                    continue

                logger.info(f"Orchestrator: Got transcript: '{result.text}' (is_final={result.is_final})")

                # Process the transcript
                await self.process_transcript(call_sid, result.text, result.language, result.is_final)

                # If call ended, stop consuming
                if context.state == ConversationState.ENDED:
                    logger.info(f"Orchestrator: Call ended, stopping transcript consumer")
                    break

        except asyncio.CancelledError:
            logger.info(f"Orchestrator: Transcript consumer cancelled for call {call_sid}")
        except Exception as e:
            logger.error(f"Orchestrator: Error in transcript consumer: {e}")
            import traceback
            logger.error(f"Orchestrator: Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"Orchestrator: Transcript consumer stopped for call {call_sid}")

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

        # ===== CALLER NAME EXTRACTION (New Callers) =====
        if not context.is_known_caller and not context.name_collected:
            extracted_name = self._extract_caller_name(transcript)
            if extracted_name:
                context.caller_name = extracted_name
                context.name_collected = True
                # Save caller info
                self.caller_service.register_caller(
                    context.phone_number,
                    extracted_name,
                    context.language
                )
                context.is_known_caller = True
                logger.info(f"Orchestrator: Extracted and saved caller name: {extracted_name}")
                # Acknowledge the name
                if context.language == "ar":
                    response = f"أهلاً {extracted_name}! كيف يمكنني مساعدتك اليوم؟"
                else:
                    response = f"Nice to meet you, {extracted_name}! How can I help you today?"
                await self._speak_to_caller(call_sid, response, context.language)
                return

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

        # Build system prompt with caller context
        system_prompt = self._system_prompt

        # Add caller name to prompt if known
        if context.caller_name:
            system_prompt += f"\n\nCALLER CONTEXT:\n- The caller's name is {context.caller_name}. Use their name naturally in your response to be warm and personal."

        # Build message list
        messages = [
            Message(role=LLMRole.SYSTEM, content=system_prompt)
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
        import traceback
        logger.info(f"Orchestrator: _speak_to_caller START - text: '{text[:50]}...'")

        context = self._conversations.get(call_sid)
        twilio_handler = self._twilio_handlers.get(call_sid)

        logger.info(f"Orchestrator: context={context is not None}, twilio_handler={twilio_handler is not None}")

        if not context or not twilio_handler:
            logger.error(f"Orchestrator: Missing context or handler - context={context}, handler={twilio_handler}")
            return

        # Check if TTS is available
        if not self.tts:
            logger.warning("TTS not available, cannot speak to caller")
            context.state = ConversationState.LISTENING
            return

        try:
            # Stream TTS
            from services.tts.tts_base import TTSRequest, TTSChunk

            request = TTSRequest(
                text=text,
                language=language,
                voice_id="Rachel"  # Can be customized per language
            )

            logger.info(f"Orchestrator: Synthesizing TTS...")
            # For now, synthesize full audio then send
            response = await self.tts.synthesize(request)
            logger.info(f"Orchestrator: TTS response - format={response.format}, sample_rate={response.sample_rate}, bytes={len(response.audio_data)}")

            # Convert MP3 audio to μ-law 8kHz for Twilio
            logger.info(f"Orchestrator: Converting audio to μ-law 8kHz...")
            from services.audio.audio_converter import convert_twilio_audio
            mulaw_audio = convert_twilio_audio(
                response.audio_data,
                input_format=response.format,
                input_rate=response.sample_rate
            )

            if not mulaw_audio:
                logger.error("Orchestrator: Audio conversion returned empty bytes")
                return

            logger.info(f"Orchestrator: Audio converted - {len(mulaw_audio)} bytes μ-law")

            # Send to Twilio in correct format
            logger.info(f"Orchestrator: Sending audio to Twilio (stream_sid={twilio_handler.stream_sid})...")
            await twilio_handler.send_audio(mulaw_audio)
            logger.info(f"Orchestrator: Audio sent to Twilio successfully")

        except Exception as e:
            logger.error(f"Orchestrator: Exception in _speak_to_caller: {e}")
            logger.error(f"Orchestrator: Traceback:\n{traceback.format_exc()}")

        finally:
            context.state = ConversationState.LISTENING
            logger.info(f"Orchestrator: _speak_to_caller END")

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

    def _extract_caller_name(self, transcript: str) -> Optional[str]:
        """
        Extract caller name from transcript

        Handles various formats:
        - "My name is John" → "John"
        - "This is Ahmed" → "Ahmed"
        - "أنا محمد" → "محمد"
        - "I'm Sarah" → "Sarah"

        Args:
            transcript: User's transcript

        Returns:
            Extracted name or None
        """
        import re

        transcript = transcript.strip()
        if not transcript:
            return None

        # English patterns
        en_patterns = [
            r"(?:my name is|i'm|i am|this is|call me|it's) (\w+)",
            r"^(\w+) here",  # "John here"
            r"speaking (\w+)",  # "This is John speaking"
        ]

        for pattern in en_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out non-name words
                if len(name) > 1 and name.lower() not in ["yes", "no", "not", "is", "it"]:
                    return name.capitalize()

        # Arabic patterns (basic)
        # "أنا أحمد" → "أحمد"
        if "أنا " in transcript or "انا " in transcript:
            parts = transcript.split()
            for i, part in enumerate(parts):
                if part in ["أنا", "انا", "اسمي"]:
                    if i + 1 < len(parts):
                        name = parts[i + 1]
                        return name

        return None

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
