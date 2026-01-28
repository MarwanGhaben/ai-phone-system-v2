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
from services.stt import create_stt_service
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

    # =====================================================
    # CONVERSATION HISTORY (Phase 1.2 fix)
    # =====================================================
    # Stores the conversation as a list of message dicts:
    # [
    #     {"role": "user", "content": "Hello"},
    #     {"role": "assistant", "content": "Hi! How can I help?"},
    #     {"role": "user", "content": "I need tax help"},
    #     {"role": "assistant", "content": "Sure, what do you need?"},
    # ]
    # This is sent to the LLM with each request for context.
    conversation_history: list = field(default_factory=list)

    # Maximum history to keep (prevent token overflow)
    MAX_HISTORY = 10  # Keep last 10 turns (20 messages)

    def add_user_message(self, message: str) -> None:
        """Add user message to history"""
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        self._trim_history()

    def add_assistant_message(self, message: str) -> None:
        """Add assistant message to history"""
        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep history within MAX_HISTORY limit"""
        if len(self.conversation_history) > self.MAX_HISTORY * 2:
            # Remove oldest messages (keep most recent)
            self.conversation_history = self.conversation_history[-(self.MAX_HISTORY * 2):]

    def get_history_for_llm(self) -> list:
        """
        Get conversation history formatted for LLM

        Returns:
            List of message dicts compatible with OpenAI API
        """
        return self.conversation_history.copy()

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
            "metadata": self.metadata,
            "conversation_length": len(self.conversation_history)
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

    ARCHITECTURE NOTES:
    - TTS and LLM are shared (stateless, thread-safe)
    - STT is created PER-CALL (to avoid queue/buffer sharing issues)
    - Each call has its own context, handler, and STT instance
    """

    def __init__(self):
        """Initialize orchestrator with shared services"""
        settings = get_settings()

        # SHARED SERVICES (stateless, thread-safe)
        self.tts = create_elevenlabs_tts(settings.model_dump())
        self.llm = create_openai_llm(settings.model_dump())

        # STT is created PER-CALL in handle_call()
        # This prevents queue/buffer sharing between calls
        self._stt_provider = settings.stt_provider
        self._stt_config = settings.model_dump()

        # Log service availability
        stt_info = {
            'whisper': 'Whisper - batch mode, supports Arabic',
            'deepgram': 'Deepgram - real-time, English-only',
            'elevenlabs': 'ElevenLabs Scribe v2 - real-time 150ms latency, supports Arabic'
        }
        logger.info(f"STT provider: {self._stt_provider} ({stt_info.get(self._stt_provider, 'unknown')})")
        if not self.tts:
            logger.warning("TTS service not available - audio responses will be limited")

        # PER-CALL STORAGE
        # Each call gets its own: context, handler, STT instance, state lock
        self._conversations: Dict[str, ConversationContext] = {}
        self._twilio_handlers: Dict[str, TwilioMediaStreamHandler] = {}
        self._call_stt_instances: Dict[str, Any] = {}  # Per-call STT instances
        self._call_state_locks: Dict[str, asyncio.Lock] = {}  # Per-call state locks

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

        return f"""You are Amal, a friendly and professional phone receptionist for Flexible Accounting (also known as iFlex Tax), a Canadian accounting firm.

YOUR IDENTITY:
- Your name is Amal
- You work at Flexible Accounting
- Be warm, natural, and human - like a real person, not a robot
- Use casual but professional tone

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

CRITICAL - RESPONSE STYLE:
- Keep responses EXTREMELY BRIEF (1-2 sentences, max 50 words)
- This is a VOICE call - long responses frustrate callers
- Get straight to the point
- Don't repeat the question back
- Don't list multiple options at once
- Ask ONE question at a time

APPOINTMENT BOOKING:
- Accountants available: {accountant_names_en}
- In Arabic: {accountant_names_ar}
- Ask: Individual or corporate client?
- Ask: Preferred accountant? (suggest from the list above)
- Ask: Preferred date/time?
- Once you have client type + date/time, use the book_appointment function to ACTUALLY create the booking
- The system will CHECK AVAILABILITY automatically before booking
- If the system returns BOOKING_UNAVAILABLE, tell the caller the requested time is not available and suggest the alternative times provided
- IMPORTANT: Do NOT tell the caller "I've booked it" unless you have called the book_appointment function and received BOOKING_SUCCESS
- If booking fails, apologize and offer to transfer to a human

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
- If caller speaks Arabic, respond in Arabic
- If caller speaks English, respond in English
- Keep Arabic responses brief and natural
- Use simple, conversational Arabic

Remember: This is a real phone call. Be CONCISE. Be helpful. Be human."""

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
            # =====================================================
            # STEP 1: Create per-call STT instance
            # =====================================================
            # CRITICAL: Each call gets its own STT instance to avoid
            # queue/buffer sharing between concurrent calls
            logger.info(f"Orchestrator: Creating per-call STT instance (provider={self._stt_provider})")
            call_stt = create_stt_service(self._stt_provider, self._stt_config)
            self._call_stt_instances[call_sid] = call_stt

            # Create per-call state lock for thread-safe state changes
            state_lock = asyncio.Lock()
            self._call_state_locks[call_sid] = state_lock

            # =====================================================
            # STEP 2: Get caller info
            # =====================================================
            logger.info(f"Orchestrator: Getting caller info for {phone_number}")
            caller_info = self.caller_service.get_caller_info(phone_number)
            caller_name = caller_info.get("name") if caller_info else None
            caller_language = caller_info.get("language", "auto") if caller_info else "auto"

            logger.info(f"Orchestrator: Caller info - name={caller_name}, language={caller_language}")

            # =====================================================
            # STEP 3: Create conversation context
            # =====================================================
            context = ConversationContext(
                call_sid=call_sid,
                phone_number=phone_number,
                language=caller_language
            )
            context.caller_name = caller_name
            context.is_known_caller = caller_name is not None
            self._conversations[call_sid] = context
            logger.info(f"Orchestrator: Conversation context created")

            # =====================================================
            # STEP 4: Create Twilio handler
            # =====================================================
            logger.info(f"Orchestrator: Creating Twilio handler")
            twilio_handler = TwilioMediaStreamHandler(call_sid, stream_sid, websocket)
            self._twilio_handlers[call_sid] = twilio_handler

            # IMPORTANT: Manually set streaming flag since we already consumed the start event in main.py
            # The handler's handle_connection() won't see the start event, so we set this manually
            twilio_handler._is_streaming = True
            twilio_handler._is_connected = True
            logger.info(f"Orchestrator: Handler configured - streaming=True, stream_sid={stream_sid}")

            # =====================================================
            # STEP 5: Connect to per-call STT
            # =====================================================
            logger.info(f"Orchestrator: Connecting to STT...")
            await call_stt.connect()
            logger.info(f"Orchestrator: STT connected")

            # Set up media handler for incoming audio
            twilio_handler.set_media_handler(self._handle_incoming_audio(call_sid))
            logger.info(f"Orchestrator: Media handler registered")

            # =====================================================
            # STEP 6: Get greeting and send it
            # =====================================================
            logger.info(f"Orchestrator: Getting greeting...")
            greeting = self.caller_service.get_greeting_for_caller(phone_number, caller_language)
            logger.info(f"Orchestrator: Greeting: '{greeting[:50]}...'")

            # Send greeting immediately (no delay - this was causing race conditions)
            # We wait for STT to be ready first
            await asyncio.sleep(0.5)  # Brief pause for audio queue to stabilize
            logger.info(f"Orchestrator: Sending greeting...")
            async with state_lock:
                context.state = ConversationState.SPEAKING
            try:
                await self._speak_to_caller(call_sid, greeting, caller_language)
                logger.info(f"Orchestrator: Greeting sent successfully")
            except Exception as e:
                logger.error(f"Orchestrator: Failed to send greeting: {e}")
            finally:
                async with state_lock:
                    context.state = ConversationState.LISTENING

            # =====================================================
            # STEP 7: Start transcript consumer for this call's STT
            # =====================================================
            logger.info(f"Orchestrator: Starting transcript consumer...")
            transcript_task = asyncio.create_task(
                self._consume_stt_transcripts(call_sid),
                name=f"transcript_consumer_{call_sid}"
            )

            # =====================================================
            # STEP 8: Handle WebSocket connection (incoming audio)
            # =====================================================
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
        It streams the audio to this call's STT instance.

        Args:
            call_sid: Call identifier
        """
        async def process_audio(audio_data: bytes):
            """Process audio chunk"""
            # Debug: Track audio chunks
            if not hasattr(process_audio, '_chunk_count'):
                process_audio._chunk_count = 0
            process_audio._chunk_count += 1
            if process_audio._chunk_count == 1 or process_audio._chunk_count % 100 == 0:
                logger.info(f"Orchestrator: process_audio called - chunk #{process_audio._chunk_count}, {len(audio_data)} bytes")

            context = self._conversations.get(call_sid)
            if not context or context.state == ConversationState.ENDED:
                logger.debug(f"Orchestrator: process_audio skipping - no context or call ended")
                return

            # Get this call's STT instance
            call_stt = self._call_stt_instances.get(call_sid)
            if not call_stt:
                logger.warning(f"Orchestrator: No STT instance for call {call_sid}")
                return

            # Get state lock for this call
            state_lock = self._call_state_locks.get(call_sid)

            # Skip if we're speaking (barge-in prevention)
            async with state_lock:
                current_state = context.state

            if current_state == ConversationState.SPEAKING:
                # Check for barge-in (user speech during AI speech)
                if await self._detect_barge_in(audio_data):
                    logger.info("Orchestrator: === BARGE-IN DETECTED === User interrupted!")
                    async with state_lock:
                        context.state = ConversationState.LISTENING

                    # Stop TTS generation
                    if self.tts:
                        await self.tts.stop()

                    # Stop audio playback on Twilio immediately
                    twilio_handler = self._twilio_handlers.get(call_sid)
                    if twilio_handler:
                        await twilio_handler.clear_audio()

                    # Feed this audio chunk to STT so the user's words aren't lost
                    from services.stt.stt_base import AudioChunk
                    await call_stt.stream_audio(AudioChunk(data=audio_data))
                    return

                # Skip STT while AI is speaking (prevents echo)
                return

            # Stream to this call's STT instance
            from services.stt.stt_base import AudioChunk
            await call_stt.stream_audio(AudioChunk(data=audio_data))
            if process_audio._chunk_count == 1:
                logger.info(f"Orchestrator: First audio chunk sent to STT")

        return process_audio

    async def _consume_stt_transcripts(self, call_sid: str) -> None:
        """
        Consume transcripts from this call's STT service queue

        This runs as a background task alongside the WebSocket event loop.
        Each call has its own STT instance, so this consumes from the correct queue.

        Args:
            call_sid: Call identifier
        """
        context = self._conversations.get(call_sid)
        if not context:
            logger.warning(f"Orchestrator: No context found for call {call_sid}")
            return

        # Get this call's STT instance
        call_stt = self._call_stt_instances.get(call_sid)
        if not call_stt:
            logger.error(f"Orchestrator: No STT instance for call {call_sid}")
            return

        logger.info(f"Orchestrator: Transcript consumer started for call {call_sid}")

        try:
            # get_transcript() returns an async iterator from THIS CALL's STT
            async for result in call_stt.get_transcript():
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

        # IMPORTANT: Ignore transcripts while AI is speaking (barge-in prevention)
        # This prevents the AI's own voice from confusing the STT
        if context.state == ConversationState.SPEAKING:
            logger.info(f"Orchestrator: Ignoring transcript while AI speaking: '{transcript}'")
            return

        logger.info(f"Orchestrator: User said: {transcript}")

        # =====================================================
        # ADD USER MESSAGE TO CONVERSATION HISTORY (Phase 1.2)
        # =====================================================
        context.add_user_message(transcript)

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

        # Get state lock for this call
        state_lock = self._call_state_locks.get(call_sid)
        if not state_lock:
            logger.error(f"Orchestrator: No state lock for call {call_sid}")
            return

        # =====================================================
        # STATE TRANSITION: LISTENING -> THINKING -> SPEAKING
        # Using lock to prevent race conditions
        # =====================================================
        async with state_lock:
            context.state = ConversationState.THINKING

        response = await self._get_ai_response(call_sid, transcript)

        context.ai_response = response
        context.turn_count += 1

        async with state_lock:
            context.state = ConversationState.SPEAKING

        # Stream response to caller (this waits for audio to complete)
        await self._speak_to_caller(call_sid, response, context.language)

        # State is set back to LISTENING inside _speak_to_caller finally block
        # Do NOT set it here to avoid race condition

    def _get_booking_tools(self) -> list:
        """Get tool definitions for OpenAI function calling"""
        return [
            {
                "name": "book_appointment",
                "description": "Book an appointment with an accountant at iFlex Tax. Call this when the user wants to schedule/book an appointment and has provided enough details (at minimum: client type).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_type": {
                            "type": "string",
                            "enum": ["individual", "corporate"],
                            "description": "Type of client - individual or corporate"
                        },
                        "accountant_name": {
                            "type": "string",
                            "description": "Preferred accountant name (e.g. Hussam Saadaldin, Rami Kahwaji, Abdul ElFarra)"
                        },
                        "date_time": {
                            "type": "string",
                            "description": "Preferred date and time in natural language (e.g. 'tomorrow at 2 PM', 'next Monday 10 AM')"
                        },
                        "customer_name": {
                            "type": "string",
                            "description": "Customer name for the booking"
                        }
                    },
                    "required": ["client_type"]
                }
            }
        ]

    async def _execute_booking(self, call_sid: str, arguments: dict) -> str:
        """
        Execute a booking via MSGraph Bookings API

        Checks availability BEFORE creating the appointment.

        Args:
            call_sid: Call identifier
            arguments: Function call arguments from LLM

        Returns:
            Result message to feed back to LLM
        """
        import json
        from datetime import datetime, timedelta
        from services.calendar.ms_bookings_service import get_calendar_service
        from services.config.accountants_service import get_accountants_service

        context = self._conversations.get(call_sid)
        calendar = get_calendar_service()
        acc_service = get_accountants_service()

        client_type = arguments.get("client_type", "individual")
        accountant_name = arguments.get("accountant_name", "")
        date_time_str = arguments.get("date_time", "")
        customer_name = arguments.get("customer_name", context.caller_name if context else "")

        logger.info(f"Orchestrator: Executing booking - type={client_type}, accountant={accountant_name}, date={date_time_str}, customer={customer_name}")

        # Check if MSGraph is configured
        if not await calendar.is_available():
            logger.error("Orchestrator: MS Bookings not configured (missing tenant_id/client_id/secret/business_id)")
            return "BOOKING_ERROR: Microsoft Bookings is not configured. Please inform the caller that booking is temporarily unavailable and offer to transfer them to a human."

        try:
            # Look up accountant
            staff_id = ""
            service_id = ""
            matched_staff_name = ""
            if accountant_name:
                acc = acc_service.get_accountant_by_name(accountant_name)
                if acc:
                    staff_id = acc.get("staff_id", "")
                    service_id = acc.get("service_id", "")
                    accountant_name = acc.get("name", accountant_name)
                    logger.info(f"Orchestrator: Found accountant {accountant_name}, staff_id={staff_id}, service_id={service_id}")

            # If no staff/service IDs configured, try to get from MSGraph
            if not staff_id or not service_id:
                # Try to get staff members from MSGraph
                staff_members = await calendar.get_staff_members()
                if staff_members:
                    # Match by name (partial match, case-insensitive)
                    if accountant_name:
                        name_lower = accountant_name.lower()
                        for staff in staff_members:
                            staff_name_lower = staff.name.lower()
                            # Check both directions for partial match
                            if name_lower in staff_name_lower or staff_name_lower in name_lower:
                                staff_id = staff.id
                                matched_staff_name = staff.name
                                logger.info(f"Orchestrator: Matched staff from MSGraph: {staff.name} (id={staff.id})")
                                break
                            # Also check first name match
                            if name_lower.split()[0] in staff_name_lower.split() if name_lower.split() else False:
                                staff_id = staff.id
                                matched_staff_name = staff.name
                                logger.info(f"Orchestrator: Matched staff by first name: {staff.name} (id={staff.id})")
                                break
                    if not staff_id and staff_members:
                        # Use first available staff
                        staff_id = staff_members[0].id
                        matched_staff_name = staff_members[0].name
                        logger.info(f"Orchestrator: Using default staff: {staff_members[0].name}")

                # Get services
                if not service_id:
                    services = await calendar.get_services()
                    if services:
                        # Match service by client type
                        for svc in services:
                            if client_type in svc.name.lower() or client_type in svc.description.lower():
                                service_id = svc.id
                                break
                        if not service_id and services:
                            service_id = services[0].id
                            logger.info(f"Orchestrator: Using default service: {services[0].name}")

            if not service_id:
                return "BOOKING_ERROR: Could not find a matching service. Please ask the caller for more details or offer to transfer them."

            # Parse date/time
            appointment_time = None
            if date_time_str:
                # Try to parse common formats
                from dateutil import parser as date_parser
                try:
                    appointment_time = date_parser.parse(date_time_str, fuzzy=True)
                    # If no date component, assume tomorrow
                    if appointment_time.date() == datetime.now().date() and "today" not in date_time_str.lower():
                        appointment_time = appointment_time.replace(
                            year=datetime.now().year,
                            month=datetime.now().month,
                            day=datetime.now().day
                        ) + timedelta(days=1)
                except Exception:
                    logger.warning(f"Orchestrator: Could not parse date: {date_time_str}")

            if not appointment_time:
                # Default to tomorrow at 10 AM
                appointment_time = datetime.now().replace(hour=10, minute=0, second=0) + timedelta(days=1)

            # =====================================================
            # CHECK AVAILABILITY BEFORE BOOKING
            # =====================================================
            logger.info(f"Orchestrator: Checking availability for staff={matched_staff_name or accountant_name}, date={appointment_time}")
            available_slots = await calendar.get_available_slots(
                service_id=service_id,
                staff_id=staff_id,
                days_ahead=7
            )

            if available_slots:
                # Check if the requested time falls within any available slot
                requested_date = appointment_time.date()
                requested_hour = appointment_time.hour
                requested_minute = appointment_time.minute

                slot_match = False
                for slot in available_slots:
                    slot_date = slot.start_time.date()
                    slot_start_hour = slot.start_time.hour
                    slot_start_min = slot.start_time.minute
                    slot_end_hour = slot.end_time.hour
                    slot_end_min = slot.end_time.minute

                    if slot_date == requested_date:
                        # Check if requested time is within this slot
                        req_minutes = requested_hour * 60 + requested_minute
                        slot_start_minutes = slot_start_hour * 60 + slot_start_min
                        slot_end_minutes = slot_end_hour * 60 + slot_end_min
                        if slot_start_minutes <= req_minutes < slot_end_minutes:
                            slot_match = True
                            break

                if not slot_match:
                    # Requested time is NOT available - find alternatives
                    same_day_slots = [s for s in available_slots if s.start_time.date() == requested_date]

                    staff_display = matched_staff_name or accountant_name or "the accountant"
                    # Full date+time format: "Friday, January 30 at 2:00 PM"
                    full_fmt = '%A, %B %d at %I:%M %p'

                    if same_day_slots:
                        # There are other slots on that day — show full date + time for each
                        alt_times = ", ".join([s.start_time.strftime(full_fmt) for s in same_day_slots[:5]])
                        logger.info(f"Orchestrator: Requested time not available. Alternatives on same day: {alt_times}")
                        return (
                            f"BOOKING_UNAVAILABLE: The requested time ({appointment_time.strftime(full_fmt)}) "
                            f"is not available for {staff_display}. "
                            f"Available slots: {alt_times}. "
                            f"Please ask the caller which of these times works for them."
                        )
                    else:
                        # No slots at all on that day — show next available with full date + time
                        next_slots = []
                        seen_days = set()
                        for s in available_slots:
                            day_key = s.start_time.date()
                            if day_key not in seen_days:
                                next_slots.append(s.start_time.strftime(full_fmt))
                                seen_days.add(day_key)
                            if len(seen_days) >= 3:
                                break

                        if next_slots:
                            alt_info = ", ".join(next_slots)
                            logger.info(f"Orchestrator: No slots on requested day. Next available: {alt_info}")
                            return (
                                f"BOOKING_UNAVAILABLE: {staff_display} has no availability on "
                                f"{appointment_time.strftime('%A, %B %d')}. "
                                f"Next available slots: {alt_info}. "
                                f"Please ask the caller if any of these work."
                            )
                        else:
                            logger.info(f"Orchestrator: No availability found at all for {staff_display}")
                            return (
                                f"BOOKING_UNAVAILABLE: {staff_display} has no available slots in the next 7 days. "
                                f"Please apologize and offer to transfer the caller to speak with someone directly."
                            )
            else:
                logger.warning("Orchestrator: Could not fetch availability slots - proceeding with booking attempt")
                # If we can't check availability (API issue), still try to book
                # The booking API itself will reject if the slot is taken

            # Create the booking
            result = await calendar.create_booking(
                service_id=service_id,
                staff_id=staff_id,
                start_time=appointment_time,
                customer_name=customer_name or "Phone Caller",
                customer_email="",  # Not collected on phone
                customer_phone=context.phone_number if context else "",
                notes=f"Booked via AI phone system. Client type: {client_type}"
            )

            if result.success:
                logger.info(f"Orchestrator: BOOKING CREATED - id={result.appointment_id}, time={result.start_time}, staff={result.staff_name}")
                return f"BOOKING_SUCCESS: Appointment booked successfully. Appointment ID: {result.appointment_id}. Time: {result.start_time}. Staff: {result.staff_name or accountant_name}. Customer: {result.customer_name}."
            else:
                logger.error(f"Orchestrator: BOOKING FAILED - {result.error_message}")
                return f"BOOKING_ERROR: Failed to create booking: {result.error_message}. Apologize and offer to transfer to a human."

        except Exception as e:
            logger.error(f"Orchestrator: Booking exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"BOOKING_ERROR: System error while booking: {str(e)}. Apologize and offer to transfer."

    async def _get_ai_response(self, call_sid: str, user_input: str) -> str:
        """
        Get AI response for user input, with function calling support for bookings.

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

        # =====================================================
        # BUILD MESSAGES WITH CONVERSATION HISTORY (Phase 1.2)
        # =====================================================
        messages = [
            Message(role=LLMRole.SYSTEM, content=system_prompt)
        ]

        # Add conversation history from context
        history = context.get_history_for_llm()
        for msg in history:
            if msg["role"] == "user":
                messages.append(Message(role=LLMRole.USER, content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(Message(role=LLMRole.ASSISTANT, content=msg["content"]))

        # Note: user_input is already in history (added in process_transcript)

        # =====================================================
        # USE FUNCTION CALLING FOR BOOKING SUPPORT
        # =====================================================
        from services.llm.llm_base import LLMRequest, LLMChunk

        tools = self._get_booking_tools()

        # First try with tools (non-streaming to detect function calls)
        request = LLMRequest(
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=False,
            tools=tools
        )

        try:
            llm_response = await self.llm.chat_with_tools(request)

            # Check if the LLM wants to call a function
            if llm_response.tool_calls:
                for tool_call in llm_response.tool_calls:
                    if tool_call["name"] == "book_appointment":
                        import json as _json
                        arguments = _json.loads(tool_call["arguments"])
                        logger.info(f"Orchestrator: LLM requested booking: {arguments}")

                        # Execute the booking
                        booking_result = await self._execute_booking(call_sid, arguments)

                        # Feed result back to LLM for a natural response
                        messages.append(Message(role=LLMRole.ASSISTANT, content=llm_response.content or ""))
                        # Add tool result as a user message (simplified approach)
                        messages.append(Message(role=LLMRole.USER, content=f"[SYSTEM: {booking_result}. Now respond naturally to the caller about the booking result. Keep it brief.]"))

                        # Get final response (streaming, no tools)
                        follow_up_request = LLMRequest(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=200,
                            stream=True
                        )
                        full_response = ""
                        async for chunk in self.llm.chat_stream(follow_up_request):
                            full_response += chunk.delta

                        response = full_response.strip()
                        context.add_assistant_message(response)
                        return response

            # No tool calls - use the direct text response
            response = llm_response.content.strip()

        except Exception as e:
            logger.error(f"Orchestrator: Function calling failed, falling back to streaming: {e}")
            # Fallback to streaming without tools
            request = LLMRequest(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                stream=True
            )
            full_response = ""
            async for chunk in self.llm.chat_stream(request):
                full_response += chunk.delta
            response = full_response.strip()

        # =====================================================
        # ADD ASSISTANT RESPONSE TO HISTORY (Phase 1.2)
        # =====================================================
        context.add_assistant_message(response)

        return response

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
        state_lock = self._call_state_locks.get(call_sid)

        logger.info(f"Orchestrator: context={context is not None}, twilio_handler={twilio_handler is not None}")

        if not context or not twilio_handler:
            logger.error(f"Orchestrator: Missing context or handler - context={context}, handler={twilio_handler}")
            return

        # Check if TTS is available
        if not self.tts:
            logger.warning("TTS not available, cannot speak to caller")
            async with state_lock:
                context.state = ConversationState.LISTENING
            return

        try:
            # Stream TTS
            from services.tts.tts_base import TTSRequest, TTSChunk

            request = TTSRequest(
                text=text,
                language=language,
                voice_id=None  # Uses default_voice_id from TTS settings (ELEVENLABS_VOICE_ID env)
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
            # =====================================================
            # STATE TRANSITION: SPEAKING -> LISTENING
            # Using lock to prevent race conditions
            # =====================================================
            async with state_lock:
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

        Uses energy-based detection with consecutive frame requirement
        to avoid false triggers from noise bursts.

        Args:
            audio_data: Audio chunk to analyze

        Returns:
            True if user is speaking (barge-in detected)
        """
        # Calculate audio energy (μ-law: silence is ~128, speech deviates from 128)
        sample_count = min(len(audio_data), 1000)
        if sample_count == 0:
            return False

        energy = sum(abs(byte - 128) for byte in audio_data[:sample_count]) / sample_count

        # Track consecutive high-energy frames to avoid false triggers
        if not hasattr(self, '_barge_in_consecutive'):
            self._barge_in_consecutive = 0

        # Threshold: 0.3 * 100 = 30 is default. Use lower value (15) for better sensitivity.
        threshold = self._energy_threshold * 50  # More sensitive (was *100)

        if energy > threshold:
            self._barge_in_consecutive += 1
            # Require 3 consecutive high-energy frames (~60ms of speech) to trigger
            if self._barge_in_consecutive >= 3:
                logger.info(f"Orchestrator: Barge-in triggered - energy={energy:.1f}, threshold={threshold:.1f}, consecutive={self._barge_in_consecutive}")
                self._barge_in_consecutive = 0
                return True
        else:
            self._barge_in_consecutive = 0

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

        # =====================================================
        # CRITICAL: Disconnect and cleanup this call's STT instance
        # =====================================================
        if call_sid in self._call_stt_instances:
            logger.info(f"Orchestrator: Disconnecting call-specific STT for {call_sid}")
            await self._call_stt_instances[call_sid].disconnect()
            del self._call_stt_instances[call_sid]

        # Clean up state lock
        if call_sid in self._call_state_locks:
            del self._call_state_locks[call_sid]

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
