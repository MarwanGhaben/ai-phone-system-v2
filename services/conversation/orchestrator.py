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

    # Pending booking (set by check_appointment, used by confirm_appointment)
    pending_booking: Optional[Dict[str, Any]] = None

    # Found appointments (set by lookup_my_bookings, used by cancel_booking)
    found_appointments: Optional[list] = None

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

        # Barge-in detection (per-call)
        self._barge_in_consecutive: Dict[str, int] = {}  # Per-call consecutive high-energy frame count
        self._barge_in_speech_start: Dict[str, float] = {}  # Per-call timestamp when SPEAKING started
        self._barge_in_reset_done: Dict[str, bool] = {}  # Per-call flag: True if barge-in already triggered STT reset

        # Echo guard: tracks when audio playback is expected to finish per call
        # During this window, STT transcripts are ignored (echo from phone speaker)
        self._echo_guard_until: Dict[str, float] = {}

        # Garbled transcript counter: tracks consecutive garbled drops per call
        # After N garbled drops, auto-reconnect STT with Arabic (the most common cause)
        self._garbled_drop_count: Dict[str, int] = {}
        self._GARBLED_AUTO_SWITCH_THRESHOLD = 1  # After 1 garbled drop, try Arabic immediately

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
        from datetime import datetime
        acc_service = get_accountants_service()
        accountant_names_en = acc_service.get_names_formatted("en")
        accountant_names_ar = acc_service.get_names_formatted("ar")

        # Inject today's date so the LLM can calculate correct dates
        today = datetime.now()
        today_str = today.strftime('%A, %B %d, %Y')  # e.g. "Friday, January 30, 2026"

        return f"""You are Sarah, a friendly and professional phone receptionist for Flexible Accounting (also known as iFlex Tax), a Canadian accounting firm.

YOUR IDENTITY:
- Your name is Sarah
- You work at Flexible Accounting
- Be warm, natural, and human - like a real person, not a robot
- Use casual but professional tone

TODAY'S DATE: {today_str}
Use this to calculate correct dates. For example if today is Friday January 30, then:
- "tomorrow" = Saturday, January 31
- "Monday" or "next Monday" = Monday, February 02
- "next week Tuesday" = Tuesday, February 03

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
- Keep responses BRIEF but NATURAL — like a real receptionist, not a robot
- This is a VOICE call — be efficient but friendly
- For simple Q&A: Keep it short (1-2 sentences max)
- If asked about services: "We handle taxes, bookkeeping, and payroll. Would you like to book an appointment?"
- Ask ONE question at a time
- In Arabic, be concise — Arabic speech takes longer to say aloud

AFTER COMPLETING A TASK (booking confirmed, appointment cancelled, question answered):
- Be warm and professional: "Is there anything else I can help you with?" / "هل هناك شيء آخر أقدر أساعدك فيه؟"
- This shows you care and gives the caller a chance to ask more or say goodbye
- If they say "no, that's all" or "thanks, bye", say a warm goodbye and end the call

BOOKING CONFIRMATIONS:
- Confirm the details clearly: "Your appointment with Hussam is booked for Monday at 2 PM. Is there anything else?"
- In Arabic: "تم حجز موعدك مع حسام يوم الاثنين الساعة الثانية ظهراً. هل تحتاج شيء آخر؟"

APPOINTMENT BOOKING (TWO-STEP PROCESS):
- Accountants available: {accountant_names_en}
- In Arabic: {accountant_names_ar}
- BUSINESS HOURS (NORTH AMERICAN SCHEDULE): Monday, Tuesday, Wednesday, Thursday, and FRIDAY are WORKDAYS. The office is OPEN on all these days including FRIDAY.
- CLOSED DAYS: ONLY Saturday (السبت) and Sunday (الأحد) are closed. These are the ONLY days the office is closed.
- CRITICAL: Friday (الجمعة/يوم الجمعة) is a WORKDAY - the office IS OPEN on Friday. Do NOT confuse this with Middle Eastern weekend schedules.
- If a caller requests Saturday or Sunday, politely let them know those are the only closed days and suggest Friday or Monday instead.
- Ask: Individual or corporate client?
- Ask: Preferred accountant? (suggest from the list above)
- Ask: Preferred date/time?
- STEP 1: Call check_appointment to check availability. NEVER call confirm_appointment without checking first.
- STEP 2: If SLOT_AVAILABLE, tell the caller the time is available and ASK them to confirm before booking. Example: "The appointment with Rami on Monday, February 02 at 2:00 PM is available. Would you like me to go ahead and book it?"
- STEP 3: ONLY after the caller says YES/confirms, call confirm_appointment with confirm=true. If they say no, call confirm_appointment with confirm=false.
- CRITICAL: date_time parameter MUST be in "YYYY-MM-DD HH:MM" format (24-hour). Use today's date ({today_str}) to calculate. NEVER pass Arabic text as date_time.
- CRITICAL: accountant_name parameter MUST be the English name (e.g. "Hussam Saadaldin", "Rami Kahwaji", "Abdul ElFarra"). NEVER pass Arabic names.
- IMPORTANT: Callers may mispronounce or approximate accountant names. Match to the closest name: "Husain"/"Hussein"/"Hosam" → "Hussam Saadaldin", "Rami"/"رامي" → "Rami Kahwaji", "Abdul"/"عبدول" → "Abdul ElFarra". NEVER say "we don't have that accountant" if the name is close to one on the list.
- IMPORTANT: SLOT_BUSY/SCHEDULE_FULL means the ACCOUNTANT is busy, NOT that the office is closed. The office is open Monday through Friday (including Friday!). ONLY Saturday and Sunday are when the office is closed.
- If the system returns SLOT_BUSY or SCHEDULE_FULL, tell the caller the accountant is busy at that time (not "office closed") and read the alternative times EXACTLY as provided.
- CRITICAL: After SLOT_BUSY/SCHEDULE_FULL, when the caller picks a new time, you MUST call check_appointment AGAIN with the new date_time. NEVER call confirm_appointment directly — always re-check the new time first.
- IMPORTANT: Do NOT tell the caller "I've booked it" unless you received BOOKING_SUCCESS from confirm_appointment.
- If booking fails, apologize and offer to transfer to a human

MODIFYING/CANCELLING EXISTING APPOINTMENTS:
- If a caller says they have an existing appointment they want to change, cancel, or check on:
  1. Call lookup_my_bookings to find their appointments (uses their phone number automatically)
  2. If appointments found, tell them what you found: "I found your appointment with [accountant] on [date/time]."
  3. Ask if they want to cancel it and book a new time
  4. If they confirm cancellation, call cancel_booking with the appointment_id and confirm_cancel=true
  5. After cancellation, offer to book a new appointment at a different time
- If no appointments found, let them know and offer to book a new one
- Example phrases that trigger this: "I need to change my appointment", "أبي أغير موعدي", "I want to reschedule", "cancel my booking"

WHEN TO TRANSFER:
- Caller EXPLICITLY asks to speak with a specific person by name (e.g. "أبي أكلم حسام" / "I want to speak to Hussam")
- Caller EXPLICITLY asks to be transferred (e.g. "حولني" / "transfer me")
- Complex tax planning questions that require an accountant's expertise
- Client-specific questions (their tax situation, past returns)
- IMPORTANT: Do NOT transfer just because you don't fully understand what the caller said.
  Instead, ask a clarifying question like "عذراً، ممكن توضح أكثر؟" / "Sorry, could you clarify?"
- IMPORTANT: Do NOT transfer if the caller's request is unclear or vague. Always ask for clarification first.
- Only transfer after you've tried at least once to understand and help the caller.

TRANSFER SCRIPT:
"Of course, let me transfer you now." (English)
"بالتأكيد، سأحولك الآن." (Arabic)

ENDING THE CALL:
- When the caller says goodbye, thanks you and is done, or clearly wants to hang up — say a brief goodbye and call the end_call tool.
- Examples: "bye" / "goodbye" / "thanks, that's all" / "مع السلامة" / "شكراً، خلص" / "يلا باي"
- IMPORTANT: Do NOT end the call just because the caller says "thank you" in the middle of a conversation. "Thanks" mid-conversation is NOT a goodbye — the caller may have more questions.
- Only end when the conversation is truly finished and the caller has no more requests.
- Always say a natural goodbye BEFORE calling end_call — never hang up silently.

LANGUAGE DETECTION:
- If caller speaks Arabic, respond in Arabic
- If caller speaks English, respond in English
- Keep Arabic responses brief and natural
- Use simple, conversational Arabic

CRITICAL - ARABIC NUMBER/TIME PRONUNCIATION:
- When speaking Arabic, ALWAYS write times and dates using Arabic words, NOT digits
- NEVER use "02:30 PM" or "10:00 AM" in Arabic responses — the voice system cannot pronounce them
- Instead say: "الساعة الثانية والنصف بعد الظهر" (2:30 PM) or "الساعة العاشرة صباحاً" (10:00 AM)
- For days use Arabic: الاثنين، الثلاثاء، الأربعاء، الخميس، الجمعة، السبت، الأحد
- For months use Arabic: يناير، فبراير، مارس، أبريل، etc.
- Write all numbers as Arabic words: واحد، اثنين، ثلاثة، etc.
- Example: instead of "Monday, February 02 at 2:30 PM", say "يوم الاثنين الثاني من فبراير الساعة الثانية والنصف بعد الظهر"

FIRST INTERACTION WITH NEW CALLERS:
- After greeting, the caller will respond (choose a language, say hello, or ask a question directly).
- If their name is unknown, ask for it in your FIRST response after they choose a language. For example in Arabic: "أهلاً! ممكن أعرف اسمك؟" or in English: "Great! And what's your name?"
- This is important — always ask for the name before helping with their request.
- If the caller says their name unprompted, register it immediately.

CALLER NAME REGISTRATION:
- CRITICAL: When the caller tells you their name, you MUST call the register_caller_name tool BEFORE responding. This is your #1 priority — never skip this tool call.
- NAME CORRECTIONS: If a caller says their name is wrong and corrects it (e.g. "اسمي غادة مش غايد" / "My name is Ghada not Gaid"), you MUST call register_caller_name with the corrected name. The caller knows their own name — always trust them.
- Only register real human names (e.g. "مروان", "Marwan", "Ahmed", "أحمد", "غادة", "Ghada")
- NEVER register garbled/unclear speech as a name. If the text looks nonsensical or garbled (random sounds, filler words like "همم", "اه"), ask the caller to repeat their name clearly.
- If the caller says "أنا اسمي مروان" → call register_caller_name with "مروان"
- If the caller says "My name is John" → call register_caller_name with "John"
- If the caller says "اسمي غادة مش غايد" → call register_caller_name with "غادة"
- If the caller says something like "أنا بحب العربي" (I prefer Arabic) → this is NOT a name, do not register anything
- A real name is typically 1-3 words and is a recognizable personal name. If it doesn't look like a name, ask again.

HANDLING UNCLEAR/GARBLED SPEECH:
- Phone calls often have audio quality issues — speech may be unclear or garbled
- If you receive garbled or unclear text, DO NOT try to interpret it as a name or meaningful input
- Simply say: "Sorry, I didn't catch that. Could you repeat?" / "عذراً، ما سمعتك منيح. ممكن تعيد؟"
- NEVER guess what the caller meant from garbled text
- If the text contains characters from unexpected scripts (Bengali, Hindi, etc.) while the conversation is in Arabic or English, treat it as garbled audio and ask to repeat

Remember: This is a real phone call. Be CONCISE. Be helpful. Be human."""

    async def _generate_greeting(self, context: ConversationContext) -> str:
        """
        Generate greeting via LLM based on caller context.
        GPT-4o decides what to say based on whether caller is known/new, their language, etc.
        Falls back to a simple template if LLM fails.
        """
        from services.llm.llm_base import LLMRequest

        try:
            if context.caller_name and context.language and context.language != "auto":
                # Known caller with language preference
                greeting_prompt = (
                    f"You are Sarah, phone receptionist at Flexible Accounting. "
                    f"A returning caller just picked up the phone. Their name is {context.caller_name} "
                    f"and they prefer {'Arabic' if context.language == 'ar' else 'English'}. "
                    f"Generate a warm, brief greeting (1 sentence max). Introduce yourself as Sarah. "
                    f"Use their name. Speak in their preferred language. Ask how you can help today. "
                    f"Example Arabic: 'مرحباً مروان! أنا سارة من فليكسبل أكاونتنغ، كيف بإمكاني مساعدتك؟' "
                    f"Example English: 'Hey Marwan! It's Sarah from Flexible Accounting. How can I help you today?'"
                )
            elif context.caller_name:
                # Known caller, unknown language
                greeting_prompt = (
                    f"You are Sarah, phone receptionist at Flexible Accounting. "
                    f"A returning caller just picked up the phone. Their name is {context.caller_name}. "
                    f"Generate a warm, brief greeting (1 sentence max) in English. Introduce yourself as Sarah. "
                    f"Use their name. Ask how you can help today. "
                    f"Example: 'Hey {context.caller_name}! It's Sarah from Flexible Accounting. How can I help you today?'"
                )
            else:
                # New caller — use hardcoded short greeting (LLM greetings are too wordy)
                # Name will be asked naturally by LLM after caller picks a language
                return "Hi, I'm Sarah from Flexible Accounting. English or Arabic?"

            request = LLMRequest(
                messages=[Message(role=LLMRole.USER, content=greeting_prompt)],
                temperature=0.8,
                max_tokens=60,
                stream=False
            )

            response = await self.llm.chat(request)
            greeting = response.content.strip().strip('"')  # Remove any quotes LLM might add

            if greeting and len(greeting) > 5:
                logger.info(f"Orchestrator: LLM generated greeting: '{greeting[:60]}...'")
                return greeting

        except Exception as e:
            logger.warning(f"Orchestrator: LLM greeting failed, using fallback: {e}")

        # Fallback — simple templates if LLM fails (always include Sarah's name)
        if context.caller_name:
            if context.language == "ar":
                return f"مرحباً {context.caller_name}! أنا سارة من فليكسبل أكاونتنغ، كيف بإمكاني مساعدتك؟"
            else:
                return f"Hey {context.caller_name}! It's Sarah from Flexible Accounting. How can I help you today?"
        else:
            return "Hi, I'm Sarah from Flexible Accounting. English or Arabic?"

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
            # =====================================================
            # STEP 1a: Get caller info FIRST (need language for STT)
            # =====================================================
            logger.info(f"Orchestrator: Getting caller info for {phone_number}")
            caller_info = await self.caller_service.get_caller_info(phone_number)
            caller_name = caller_info.get("name") if caller_info else None
            caller_language = caller_info.get("language", "auto") if caller_info else "auto"

            logger.info(f"Orchestrator: Caller info - name={caller_name}, language={caller_language}")

            # =====================================================
            # STEP 1b: Create per-call STT with language hint
            # =====================================================
            # ALWAYS set STT language explicitly — auto-detect is unreliable on phone audio.
            # For known callers: use their stored language preference.
            # For new callers: start with Arabic (primary business language).
            #   - Arabic STT can still detect "English"/"انجلش" keywords for language switching
            #   - Auto-detect drops short Arabic words like "عربي" entirely
            stt_config = self._stt_config.copy()
            if caller_language and caller_language != "auto":
                stt_config['elevenlabs_stt_language'] = caller_language
                logger.info(f"Orchestrator: Setting STT language to '{caller_language}' for known caller")
            else:
                # New caller: start with Arabic instead of unreliable auto-detect
                stt_config['elevenlabs_stt_language'] = "ar"
                logger.info(f"Orchestrator: Setting STT language to 'ar' for new caller (Arabic-first business)")

            logger.info(f"Orchestrator: Creating per-call STT instance (provider={self._stt_provider})")
            call_stt = create_stt_service(self._stt_provider, stt_config)
            self._call_stt_instances[call_sid] = call_stt

            # Create per-call state lock for thread-safe state changes
            state_lock = asyncio.Lock()
            self._call_state_locks[call_sid] = state_lock

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
            # STEP 3b: Log call start to database
            # =====================================================
            try:
                from services.database import get_db_pool
                pool = await get_db_pool()
                await pool.execute(
                    """
                    INSERT INTO call_logs (call_sid, phone_number, caller_name, language, started_at, status)
                    VALUES ($1, $2, $3, $4, NOW(), 'in_progress')
                    ON CONFLICT (call_sid) DO NOTHING
                    """,
                    call_sid,
                    phone_number,
                    caller_name or "Unknown",
                    caller_language or "en"
                )
                logger.info(f"Orchestrator: Call logged to database")
            except Exception as db_err:
                logger.warning(f"Orchestrator: Failed to log call start to DB: {db_err}")

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
            # STEP 6: Start connection handler + transcript consumer
            # =====================================================
            # IMPORTANT: Start these BEFORE the greeting so that:
            # 1. The audio queue consumer is active (sends queued audio to Twilio)
            # 2. Barge-in detection works during greeting playback
            # 3. Transcript consumer is ready for caller's first response
            logger.info(f"Orchestrator: Starting connection handler...")
            connection_task = asyncio.create_task(
                twilio_handler.handle_connection(),
                name=f"connection_{call_sid}"
            )
            logger.info(f"Orchestrator: Starting transcript consumer...")
            transcript_task = asyncio.create_task(
                self._consume_stt_transcripts(call_sid),
                name=f"transcript_consumer_{call_sid}"
            )
            # Brief pause for connection handler to initialize
            await asyncio.sleep(0.2)

            # =====================================================
            # STEP 7: Generate greeting via LLM and send it
            # =====================================================
            # GPT-4o generates the greeting based on caller context
            # (returning vs new caller, language preference, etc.)
            # Fallback to simple template if LLM fails.
            logger.info(f"Orchestrator: Generating LLM greeting...")
            greeting = await self._generate_greeting(context)
            greeting_lang = caller_language if caller_language != "auto" else "en"
            logger.info(f"Orchestrator: Greeting: '{greeting[:80]}...'")

            # Add greeting to conversation history so LLM knows what it already said
            context.add_assistant_message(greeting)

            # Send greeting (blocks until playback completes — barge-in active)
            logger.info(f"Orchestrator: Sending greeting...")
            import time
            async with state_lock:
                context.state = ConversationState.SPEAKING
            self._barge_in_speech_start[call_sid] = time.time()
            try:
                await self._speak_to_caller(call_sid, greeting, greeting_lang)
                logger.info(f"Orchestrator: Greeting playback complete")
            except Exception as e:
                logger.error(f"Orchestrator: Failed to send greeting: {e}")
                # Ensure we're in LISTENING state on error
                async with state_lock:
                    context.state = ConversationState.LISTENING

            # =====================================================
            # STEP 8: Wait for connection to end (call hangup)
            # =====================================================
            logger.info(f"Orchestrator: Waiting for call to end...")
            try:
                await connection_task
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
                if await self._detect_barge_in(audio_data, call_sid):
                    logger.info("Orchestrator: === BARGE-IN DETECTED === User interrupted!")
                    async with state_lock:
                        context.state = ConversationState.LISTENING

                    # Clear echo guard so the caller's speech isn't blocked
                    self._echo_guard_until[call_sid] = 0

                    # Stop TTS generation
                    if self.tts:
                        await self.tts.stop()

                    # Stop audio playback on Twilio immediately
                    twilio_handler = self._twilio_handlers.get(call_sid)
                    if twilio_handler:
                        await twilio_handler.clear_audio()

                    # Reset STT for clean listening after barge-in
                    await call_stt.reset_for_listening()
                    # Mark that barge-in already did the reset (prevent double reset in _speak_to_caller)
                    self._barge_in_reset_done[call_sid] = True
                else:
                    # SPEAKING and no barge-in: Do NOT feed audio to STT.
                    # The AI's voice comes back through the phone mic as echo.
                    # Feeding this echo to STT pollutes its VAD buffer, causing
                    # it to miss the user's first words after AI finishes speaking.
                    # STT will be reset with a clean connection when we transition
                    # to LISTENING (see _speak_to_caller).
                    return

            # LISTENING state: feed audio to STT
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

        # Track last processed final transcript to deduplicate
        # ElevenLabs sends both committed_transcript and final_transcript
        # for the same utterance, causing duplicate processing.
        last_final_text = None

        try:
            # Outer loop: survives STT resets (disconnect + reconnect).
            # When reset_for_listening() disconnects STT, get_transcript() ends
            # because _is_listening becomes False. Without this outer loop,
            # the consumer dies and no transcripts are ever processed again.
            while True:
                # Check if call is still active
                if context.state == ConversationState.ENDED:
                    logger.info(f"Orchestrator: Call ended, stopping transcript consumer")
                    break

                # Re-fetch STT instance (same object, but its internal queue
                # is recreated on reconnect)
                call_stt = self._call_stt_instances.get(call_sid)
                if not call_stt:
                    logger.info(f"Orchestrator: STT instance removed, stopping consumer")
                    break

                try:
                    async for result in call_stt.get_transcript():
                        if not result or not result.text:
                            continue

                        # Deduplicate final transcripts
                        if result.is_final:
                            if result.text == last_final_text:
                                logger.info(f"Orchestrator: Skipping duplicate transcript: '{result.text[:50]}'")
                                continue
                            last_final_text = result.text

                        logger.info(f"Orchestrator: Got transcript: '{result.text}' (is_final={result.is_final})")
                        await self.process_transcript(call_sid, result.text, result.language, result.is_final)

                        if context.state == ConversationState.ENDED:
                            break

                except asyncio.CancelledError:
                    raise  # Propagate cancellation
                except Exception as e:
                    logger.warning(f"Orchestrator: Transcript iterator error (will retry): {e}")

                # get_transcript() ended — this should rarely happen now since
                # reset_for_listening() keeps _is_listening=True.
                # Use exponential backoff to avoid spin-looping if something is wrong.
                if context.state != ConversationState.ENDED:
                    logger.warning(f"Orchestrator: Transcript iterator ended unexpectedly, retrying in 1s...")
                    await asyncio.sleep(1.0)

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

        # Only process final transcripts
        if not is_final:
            return

        # IMPORTANT: Ignore transcripts while AI is speaking (barge-in prevention)
        # This prevents the AI's own voice from confusing the STT
        if context.state == ConversationState.SPEAKING:
            logger.info(f"Orchestrator: Ignoring transcript while AI speaking: '{transcript}'")
            return

        # ECHO GUARD: Ignore transcripts during audio playback window
        # Even after state goes to LISTENING, audio is still playing through
        # the phone speaker. The caller's phone mic picks this up as echo,
        # which STT transcribes as garbled text (e.g., Ol Chiki script).
        import time
        echo_guard_end = self._echo_guard_until.get(call_sid, 0)
        if time.time() < echo_guard_end:
            remaining = echo_guard_end - time.time()
            logger.info(f"Orchestrator: Ignoring transcript during echo guard ({remaining:.1f}s remaining): '{transcript[:50]}'")
            return

        logger.info(f"Orchestrator: User said: {transcript}")

        # =====================================================
        # LANGUAGE DETECTION (auto-detect from first response)
        # =====================================================
        # Arabic characters are a DEFINITIVE signal — always check first,
        # because STT may return language="en" even for Arabic text
        import re
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', transcript))

        # Detect garbled text: WHITELIST approach.
        # This system only handles English (Latin) and Arabic.
        # If the transcript contains characters outside these scripts
        # (plus common punctuation/digits), it's garbled STT output.
        # Strip allowed characters and check if anything remains.
        import re
        allowed_chars = re.sub(
            r'[\u0000-\u007F'    # Basic Latin (ASCII — letters, digits, punctuation)
            r'\u00A0-\u00FF'     # Latin-1 Supplement (accented chars)
            r'\u0100-\u024F'     # Latin Extended
            r'\u0600-\u06FF'     # Arabic
            r'\u0750-\u077F'     # Arabic Supplement
            r'\u08A0-\u08FF'     # Arabic Extended-A
            r'\uFB50-\uFDFF'     # Arabic Presentation Forms-A
            r'\uFE70-\uFEFF'     # Arabic Presentation Forms-B
            r'\u200B-\u200F'     # Zero-width and directional marks
            r'\u2000-\u206F'     # General punctuation
            r'\s]+',             # Whitespace
            '',
            transcript
        )
        has_unexpected_script = len(allowed_chars) > 0

        if has_unexpected_script and not has_arabic:
            # STT garbled the audio into an unexpected script.
            # This is usually echo or misrecognized Arabic on phone-quality audio.
            logger.warning(f"Orchestrator: Detected non-Latin/Arabic script in transcript: '{transcript}' (foreign chars: '{allowed_chars[:30]}') — ignoring")

            # Track consecutive garbled drops for this call
            self._garbled_drop_count[call_sid] = self._garbled_drop_count.get(call_sid, 0) + 1
            garbled_count = self._garbled_drop_count[call_sid]
            logger.info(f"Orchestrator: Garbled drop count for {call_sid}: {garbled_count}")

            # After N garbled drops, the caller is almost certainly speaking Arabic
            # but STT auto-detect is failing. Auto-switch to Arabic and re-prompt.
            if garbled_count == self._GARBLED_AUTO_SWITCH_THRESHOLD and context.language == "auto":
                logger.warning(f"Orchestrator: {garbled_count} consecutive garbled transcripts — auto-switching to Arabic")
                context.language = "ar"
                await self._reconnect_stt_with_language(call_sid, "ar", force=True)
                # Re-prompt the caller so they know the system is still listening
                await self._speak_to_caller(call_sid, "عذراً، ممكن تعيد من فضلك؟", "ar")

            return

        # LANGUAGE KEYWORD DETECTION: If the caller says "Arabic" / "arabi" etc.
        # in English, they're requesting Arabic — NOT speaking English.
        # This must be checked BEFORE other language detection logic.
        #
        # Two detection modes:
        # 1. SHORT phrases (<=5 words): keyword match (e.g. "Arabic please", "عربي")
        # 2. LONGER sentences: explicit switch PHRASES that unambiguously request
        #    a language change (e.g. "I want to talk in Arabic", "can you speak Arabic")
        #    This avoids false positives on "I'm an English speaker" while still
        #    catching "switch to Arabic" or "talk to me in Arabic".
        transcript_lower = transcript.strip().lower().rstrip('.,!?؟')
        word_count = len(transcript_lower.split())
        arabic_keywords = ["arabic", "arabi", "arabik", "3arabi", "عربي"]
        english_keywords = ["english", "inglish", "inglizi", "انجليزي", "انكليزي", "انجليش", "انجلش"]
        is_arabic_request = any(kw in transcript_lower for kw in arabic_keywords)
        is_english_request = any(kw in transcript_lower for kw in english_keywords)

        # Explicit switch phrases — unambiguous even in long sentences
        arabic_switch_phrases = [
            "speak arabic", "talk arabic", "talk in arabic", "speak in arabic",
            "switch to arabic", "change to arabic", "in arabic please",
            "can you speak arabic", "want arabic", "prefer arabic",
            "let's speak arabic", "let's talk arabic", "use arabic",
            "respond in arabic", "answer in arabic", "reply in arabic",
            "تكلم عربي", "كلمني عربي", "حكي عربي", "بالعربي",
        ]
        english_switch_phrases = [
            "speak english", "talk english", "talk in english", "speak in english",
            "switch to english", "change to english", "in english please",
            "can you speak english", "want english", "prefer english",
            "let's speak english", "let's talk english", "use english",
            "respond in english", "answer in english", "reply in english",
            "تكلم انجليزي", "كلمني انجليزي", "بالانجليزي",
        ]
        has_arabic_switch_phrase = any(phrase in transcript_lower for phrase in arabic_switch_phrases)
        has_english_switch_phrase = any(phrase in transcript_lower for phrase in english_switch_phrases)

        # Treat as language switch if: short phrase OR explicit switch phrase OR language not yet set
        is_language_switch = (word_count <= 5 or has_arabic_switch_phrase or has_english_switch_phrase or context.language == "auto")

        if is_arabic_request and not is_english_request and is_language_switch:
            # Caller is requesting Arabic language (short phrase like "Arabic" or "عربي")
            prev_language = context.language
            context.language = "ar"
            logger.info(f"Orchestrator: Detected Arabic language REQUEST from keyword: '{transcript}'")
            # Reset garbled counter since we got a valid signal
            self._garbled_drop_count[call_sid] = 0
            # Skip reconnect if STT already started in Arabic (new callers start in ar)
            if prev_language not in ("ar", "auto"):
                await self._reconnect_stt_with_language(call_sid, "ar", force=True)
        elif is_english_request and is_language_switch:
            # Caller explicitly requested English (short phrase like "English please")
            # Check BEFORE has_arabic because "انجليش" contains Arabic chars but means "English"
            prev_language = context.language
            context.language = "en"
            logger.info(f"Orchestrator: Detected English language REQUEST from keyword: '{transcript}'")
            self._garbled_drop_count[call_sid] = 0
            if prev_language != "en":
                await self._reconnect_stt_with_language(call_sid, "en", force=True)
        elif has_arabic:
            prev_language = context.language
            context.language = "ar"
            logger.info(f"Orchestrator: Detected Arabic from text content")
            # Reset garbled counter since we got a valid Arabic transcript
            self._garbled_drop_count[call_sid] = 0
            # Reconnect STT with Arabic if switching from a different language
            # Skip reconnect if prev was "auto" and STT already started in Arabic
            if prev_language not in ("ar", "auto"):
                await self._reconnect_stt_with_language(call_sid, "ar", force=True)
        elif context.language == "auto":
            # Check for romanized Arabic (STT auto-detect often romanizes Arabic speech)
            romanized_arabic_patterns = [
                "allah", "inshallah", "yalla", "habibi", "habibti", "shukran",
                "marhaba", "ahlan", "salam", "wallah", "khalas", "mashallah",
                "bilawasamat", "bukra", "sabah", "masa", "aiwa", "naam",
                "mumkin", "arabi", "araby",
            ]
            transcript_words = transcript_lower.split()
            is_romanized_arabic = any(
                any(pat in word for pat in romanized_arabic_patterns)
                for word in transcript_words
            )

            if is_romanized_arabic:
                # Romanized Arabic detected — switch to Arabic STT
                context.language = "ar"
                logger.info(f"Orchestrator: Detected romanized Arabic in: '{transcript}' — switching to Arabic")
                self._garbled_drop_count[call_sid] = 0
                await self._reconnect_stt_with_language(call_sid, "ar", force=True)
                # Re-prompt in Arabic
                await self._speak_to_caller(call_sid, "عذراً، ممكن تعيد من فضلك؟", "ar")
                return
            elif language and language != "auto" and language == "en":
                # STT explicitly detected English — trust it
                context.language = "en"
                logger.info(f"Orchestrator: Auto-detected English from STT")
                await self._reconnect_stt_with_language(call_sid, "en")
            else:
                # Unknown or unclear language — default to Arabic (primary business language)
                # If they're speaking English, next transcript will be clearly English
                context.language = "ar"
                logger.info(f"Orchestrator: Defaulting to Arabic (primary business language)")
                await self._reconnect_stt_with_language(call_sid, "ar", force=True)

        # Reset garbled counter on any successful transcript
        if call_sid in self._garbled_drop_count and not has_unexpected_script:
            self._garbled_drop_count[call_sid] = 0

        # =====================================================
        # ADD USER MESSAGE TO CONVERSATION HISTORY (Phase 1.2)
        # =====================================================
        context.add_user_message(transcript)

        # NOISE FILTER: Ignore meaningless utterances (hesitation sounds).
        # IMPORTANT: Do NOT filter by length — Arabic words like "نعم" (yes, 3 chars)
        # and "لا" (no, 2 chars) are short but critically meaningful.
        # Use an explicit noise list instead.
        cleaned = transcript.strip().strip('.,!?؟').strip()
        noise_words = {
            # English hesitation sounds
            "eee", "ee", "uh", "um", "ah", "eh", "hmm", "hm",
            "uhh", "umm", "ahh", "ehh", "mmm", "mm", "mhm",
            "ugh", "oh", "ooh", "err", "er",
            # Arabic hesitation sounds
            "اه", "هم", "ام", "آه", "امم", "اهه",
        }
        is_noise = cleaned.lower() in noise_words
        if is_noise:
            logger.info(f"Orchestrator: Ignoring noise/hesitation: '{transcript}'")
            return

        # All transcripts go to GPT-4o — the LLM decides everything (greeting follow-up,
        # name collection, booking, transfer, goodbye, etc.). No hardcoded responses.
        # via function calling tools. No hardcoded keyword matching.

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

        # Check if transfer was already handled (don't speak the marker)
        if response == "__TRANSFER_HANDLED__":
            async with state_lock:
                context.state = ConversationState.LISTENING
            return

        context.ai_response = response
        context.turn_count += 1

        import time
        async with state_lock:
            context.state = ConversationState.SPEAKING
        self._barge_in_speech_start[call_sid] = time.time()

        # Stream response to caller (this waits for audio to complete)
        await self._speak_to_caller(call_sid, response, context.language)

        # State is set back to LISTENING inside _speak_to_caller finally block
        # Do NOT set it here to avoid race condition

    def _get_booking_tools(self) -> list:
        """Get tool definitions for OpenAI function calling"""
        return [
            {
                "name": "check_appointment",
                "description": "Check if a requested appointment slot is available. Call this FIRST when the user wants to book. Do NOT call confirm_appointment until the caller explicitly confirms.",
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
                            "description": "Preferred accountant name in ENGLISH only. Must be one of: Hussam Saadaldin, Rami Kahwaji, Abdul ElFarra"
                        },
                        "date_time": {
                            "type": "string",
                            "description": "Preferred date and time in YYYY-MM-DD HH:MM format (24-hour). Example: '2026-02-02 15:00'. Calculate the correct date based on today's date. NEVER pass Arabic text."
                        },
                        "customer_name": {
                            "type": "string",
                            "description": "Customer name for the booking"
                        }
                    },
                    "required": ["client_type"]
                }
            },
            {
                "name": "confirm_appointment",
                "description": "Confirm and finalize a booking AFTER the caller has explicitly said yes to the proposed time slot. Only call this after check_appointment returned SLOT_AVAILABLE and the caller confirmed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Must be true to confirm the booking"
                        }
                    },
                    "required": ["confirm"]
                }
            },
            {
                "name": "register_caller_name",
                "description": "Register or update the caller's name. Call this when a caller tells you their name for the first time OR corrects their name. Always trust the caller's own statement of their name over what's on file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "The caller's personal name (first name or full name). Must be an actual human name. Examples: 'Marwan', 'مروان', 'Ahmed', 'أحمد'. Never pass common words."
                        }
                    },
                    "required": ["caller_name"]
                }
            },
            {
                "name": "transfer_to_human",
                "description": "Transfer the call to a human staff member. ONLY call this when the caller EXPLICITLY asks to speak with a person by name or explicitly requests a transfer. Do NOT call this for vague or unclear requests — ask a clarifying question instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the transfer"
                        }
                    },
                    "required": ["reason"]
                }
            },
            {
                "name": "end_call",
                "description": "End the phone call. Call this AFTER you've said goodbye to the caller. Use when: the caller says goodbye/bye/thanks and is done, OR the caller explicitly asks to hang up. Do NOT end the call just because the caller said 'thank you' mid-conversation — only when the conversation is truly finished.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for ending (e.g. 'caller said goodbye', 'conversation complete')"
                        }
                    },
                    "required": ["reason"]
                }
            },
            {
                "name": "lookup_my_bookings",
                "description": "Look up the caller's existing upcoming appointments. Call this when the caller says they have an existing booking they want to change, cancel, or check on. Uses the caller's phone number to find their appointments.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "cancel_booking",
                "description": "Cancel an existing appointment. Call this ONLY after lookup_my_bookings found the caller's appointment AND the caller explicitly confirmed they want to cancel it. The system automatically uses the appointment found by lookup_my_bookings. After cancellation, offer to book a new appointment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirm_cancel": {
                            "type": "boolean",
                            "description": "Must be true to confirm cancellation"
                        }
                    },
                    "required": ["confirm_cancel"]
                }
            }
        ]

    async def _check_booking(self, call_sid: str, arguments: dict) -> str:
        """
        Check availability for a requested appointment slot.
        If available, stores pending booking data on the context for later confirmation.

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

        logger.info(f"Orchestrator: Checking booking - type={client_type}, accountant={accountant_name}, date={date_time_str}, customer={customer_name}")

        # Check if MSGraph is configured
        if not await calendar.is_available():
            logger.error("Orchestrator: MS Bookings not configured")
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
                staff_members = await calendar.get_staff_members()
                if staff_members:
                    if accountant_name:
                        name_lower = accountant_name.lower()
                        for staff in staff_members:
                            staff_name_lower = staff.name.lower()
                            if name_lower in staff_name_lower or staff_name_lower in name_lower:
                                staff_id = staff.id
                                matched_staff_name = staff.name
                                logger.info(f"Orchestrator: Matched staff from MSGraph: {staff.name} (id={staff.id})")
                                break
                            if name_lower.split()[0] in staff_name_lower.split() if name_lower.split() else False:
                                staff_id = staff.id
                                matched_staff_name = staff.name
                                logger.info(f"Orchestrator: Matched staff by first name: {staff.name} (id={staff.id})")
                                break
                    if not staff_id and staff_members:
                        staff_id = staff_members[0].id
                        matched_staff_name = staff_members[0].name
                        logger.info(f"Orchestrator: Using default staff: {staff_members[0].name}")

                if not service_id:
                    services = await calendar.get_services()
                    if services:
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
                import re
                iso_match = re.match(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})', date_time_str)
                if iso_match:
                    try:
                        appointment_time = datetime(
                            int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)),
                            int(iso_match.group(4)), int(iso_match.group(5))
                        )
                        logger.info(f"Orchestrator: Parsed ISO date: {appointment_time}")
                    except ValueError as e:
                        logger.warning(f"Orchestrator: Invalid ISO date values: {date_time_str} - {e}")

                if not appointment_time:
                    from dateutil import parser as date_parser
                    try:
                        appointment_time = date_parser.parse(date_time_str, fuzzy=True)
                        logger.info(f"Orchestrator: Parsed with dateutil: {appointment_time}")
                        if appointment_time < datetime.now():
                            logger.warning(f"Orchestrator: Parsed date {appointment_time} is in the past, adjusting to next week")
                            appointment_time += timedelta(days=7)
                    except Exception:
                        logger.warning(f"Orchestrator: Could not parse date: {date_time_str}")

            if not appointment_time:
                appointment_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
                logger.warning(f"Orchestrator: Using default appointment time: {appointment_time}")

            logger.info(f"Orchestrator: Final appointment time: {appointment_time} (from input: '{date_time_str}')")

            # =====================================================
            # CHECK AVAILABILITY
            # =====================================================
            staff_display = matched_staff_name or accountant_name or "the accountant"
            full_fmt = '%A, %B %d at %I:%M %p'

            logger.info(f"Orchestrator: Checking availability for staff={staff_display}, date={appointment_time}")
            available_slots = await calendar.get_available_slots(
                service_id=service_id,
                staff_id=staff_id,
                days_ahead=7
            )

            if available_slots:
                requested_date = appointment_time.date()
                requested_hour = appointment_time.hour
                requested_minute = appointment_time.minute

                slot_match = False
                for slot in available_slots:
                    slot_date = slot.start_time.date()
                    if slot_date == requested_date:
                        req_minutes = requested_hour * 60 + requested_minute
                        slot_start_minutes = slot.start_time.hour * 60 + slot.start_time.minute
                        slot_end_minutes = slot.end_time.hour * 60 + slot.end_time.minute
                        if slot_start_minutes <= req_minutes < slot_end_minutes:
                            slot_match = True
                            break

                if not slot_match:
                    # Not available — store partial booking so confirm_appointment
                    # can still work if LLM skips re-checking
                    context.pending_booking = {
                        "service_id": service_id,
                        "staff_id": staff_id,
                        "staff_name": matched_staff_name or accountant_name,
                        "appointment_time": None,  # No confirmed time yet
                        "customer_name": customer_name,
                        "client_type": client_type,
                        "awaiting_recheck": True,
                    }

                    same_day_slots = [s for s in available_slots if s.start_time.date() == requested_date]

                    if same_day_slots:
                        alt_times = ", ".join([s.start_time.strftime(full_fmt) for s in same_day_slots[:2]])
                        logger.info(f"Orchestrator: Requested time not available. Alternatives: {alt_times}")
                        return (
                            f"SLOT_BUSY: {staff_display} is busy at that exact time (the office IS open, just that time slot is taken). "
                            f"Same-day alternatives: {alt_times}. "
                            f"Ask which works. Call check_appointment AGAIN with new time."
                        )
                    else:
                        next_slots = []
                        seen_days = set()
                        for s in available_slots:
                            day_key = s.start_time.date()
                            if day_key not in seen_days:
                                next_slots.append(s.start_time.strftime(full_fmt))
                                seen_days.add(day_key)
                            if len(seen_days) >= 2:
                                break

                        if next_slots:
                            alt_info = ", ".join(next_slots)
                            logger.info(f"Orchestrator: No slots on requested day. Next available: {alt_info}")
                            return (
                                f"SCHEDULE_FULL: {staff_display}'s schedule is full that day (office IS open Monday-Friday, but this accountant is booked). "
                                f"Next available: {alt_info}. "
                                f"Ask which works. Call check_appointment AGAIN with new time."
                            )
                        else:
                            return (
                                f"NO_AVAILABILITY: {staff_display} has no available slots in the next 7 days. "
                                f"Apologize and offer to transfer or suggest a different accountant."
                            )

            # =====================================================
            # SLOT IS AVAILABLE — store pending booking, ask for confirmation
            # =====================================================
            context.pending_booking = {
                "service_id": service_id,
                "staff_id": staff_id,
                "staff_name": matched_staff_name or accountant_name,
                "appointment_time": appointment_time,
                "customer_name": customer_name,
                "client_type": client_type,
            }
            display_time = appointment_time.strftime(full_fmt)
            logger.info(f"Orchestrator: Slot available - pending confirmation for {display_time} with {staff_display}")

            return (
                f"SLOT_AVAILABLE: The appointment with {staff_display} on {display_time} is available. "
                f"Ask the caller to confirm: do they want to go ahead and book this appointment? "
                f"Do NOT book yet — wait for the caller to say yes."
            )

        except Exception as e:
            logger.error(f"Orchestrator: Check booking exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"BOOKING_ERROR: System error while checking availability: {str(e)}. Apologize and offer to transfer."

    async def _confirm_booking(self, call_sid: str, arguments: dict) -> str:
        """
        Confirm and create a previously checked appointment.

        Args:
            call_sid: Call identifier
            arguments: Function call arguments (must have confirm=True)

        Returns:
            Result message to feed back to LLM
        """
        from services.calendar.ms_bookings_service import get_calendar_service

        context = self._conversations.get(call_sid)
        if not context or not context.pending_booking:
            logger.warning("Orchestrator: confirm_appointment called but no pending booking")
            return (
                "BOOKING_NEEDS_RECHECK: No confirmed time slot to book. "
                "You must call check_appointment with the caller's chosen date_time first, "
                "then call confirm_appointment after SLOT_AVAILABLE."
            )

        if not arguments.get("confirm", False):
            # Caller said no — clear pending booking
            context.pending_booking = None
            return "BOOKING_CANCELLED: The caller chose not to book. Ask if they would like a different time or anything else."

        pending = context.pending_booking

        # If pending booking is from an UNAVAILABLE check (no confirmed time),
        # the LLM skipped re-checking — tell it to call check_appointment first
        if pending.get("awaiting_recheck") or pending.get("appointment_time") is None:
            logger.warning("Orchestrator: confirm_appointment called but pending booking has no confirmed time (awaiting recheck)")
            return (
                "BOOKING_NEEDS_RECHECK: The previous time was unavailable and no new time was checked. "
                "You must call check_appointment with the caller's chosen date_time first, "
                "then call confirm_appointment after SLOT_AVAILABLE."
            )

        calendar = get_calendar_service()

        # Use the latest caller name from context (may have been registered after check_appointment)
        customer_name = context.caller_name or pending.get("customer_name") or "Phone Caller"

        try:
            result = await calendar.create_booking(
                service_id=pending["service_id"],
                staff_id=pending["staff_id"],
                start_time=pending["appointment_time"],
                customer_name=customer_name,
                customer_email="",
                customer_phone=context.phone_number or "",
                notes=f"Booked via AI phone system. Client type: {pending['client_type']}"
            )

            if result.success:
                logger.info(f"Orchestrator: BOOKING CREATED - id={result.appointment_id}, time={result.start_time}, staff={result.staff_name}")

                # Save booking to local database for dashboard
                try:
                    from services.database import get_db_pool
                    pool = await get_db_pool()
                    await pool.execute(
                        """
                        INSERT INTO bookings (
                            call_sid, phone_number, client_name, client_email,
                            accountant_name, appointment_time, client_type,
                            language, status, ms_booking_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT DO NOTHING
                        """,
                        call_sid,
                        context.phone_number or "",
                        customer_name,
                        "",  # email
                        result.staff_name or pending.get("staff_name", ""),
                        pending["appointment_time"],
                        pending.get("client_type", "unknown"),
                        context.language or "en",
                        "confirmed",
                        result.appointment_id or ""
                    )
                    logger.info(f"Orchestrator: Booking saved to dashboard database")
                except Exception as db_err:
                    logger.warning(f"Orchestrator: Failed to save booking to dashboard DB: {db_err}")

                # Send SMS confirmation + schedule 24hr reminder
                display_staff = result.staff_name or pending["staff_name"] or "your accountant"
                display_time = pending["appointment_time"].strftime('%A, %B %d at %I:%M %p')
                display_customer = customer_name if customer_name != "Phone Caller" else "there"
                caller_lang = context.language if context else "en"

                asyncio.create_task(self._send_booking_sms(
                    phone_number=context.phone_number,
                    customer_name=display_customer,
                    staff_name=display_staff,
                    appointment_time_str=display_time,
                    appointment_dt=pending["appointment_time"],
                    language=caller_lang,
                ))

                # Clear pending booking
                context.pending_booking = None

                return f"BOOKING_SUCCESS: Appointment booked successfully. Appointment ID: {result.appointment_id}. Time: {result.start_time}. Staff: {result.staff_name or pending['staff_name']}. Customer: {result.customer_name}."
            else:
                logger.error(f"Orchestrator: BOOKING FAILED - {result.error_message}")
                context.pending_booking = None
                return f"BOOKING_ERROR: Failed to create booking: {result.error_message}. Apologize and offer to transfer to a human."

        except Exception as e:
            logger.error(f"Orchestrator: Confirm booking exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            context.pending_booking = None
            return f"BOOKING_ERROR: System error while booking: {str(e)}. Apologize and offer to transfer."

    async def _send_booking_sms(
        self,
        phone_number: str,
        customer_name: str,
        staff_name: str,
        appointment_time_str: str,
        appointment_dt,
        language: str,
    ) -> None:
        """
        Send booking confirmation SMS and schedule a 24hr reminder.
        Runs as a fire-and-forget background task so it doesn't block the call.
        """
        from services.sms.telnyx_sms_service import get_sms_service
        from services.sms.reminder_scheduler import get_reminder_scheduler

        sms = get_sms_service()
        if not sms.is_available():
            logger.warning("Orchestrator: SMS not configured - skipping confirmation")
            return

        # 1. Send immediate confirmation
        try:
            sent = await sms.send_booking_confirmation(
                to_number=phone_number,
                customer_name=customer_name,
                staff_name=staff_name,
                appointment_time=appointment_time_str,
                language=language,
            )
            if sent:
                logger.info(f"Orchestrator: Booking confirmation SMS sent to {phone_number}")
            else:
                logger.warning(f"Orchestrator: Failed to send confirmation SMS to {phone_number}")
        except Exception as e:
            logger.error(f"Orchestrator: SMS confirmation error: {e}")

        # 2. Schedule 24hr reminder
        try:
            scheduler = get_reminder_scheduler()
            scheduler.schedule_reminder(
                phone_number=phone_number,
                customer_name=customer_name,
                staff_name=staff_name,
                appointment_time_str=appointment_time_str,
                appointment_dt=appointment_dt,
                language=language,
            )
            logger.info(f"Orchestrator: 24hr reminder scheduled for {phone_number}")
        except Exception as e:
            logger.error(f"Orchestrator: Reminder scheduling error: {e}")

    async def _lookup_my_bookings(self, call_sid: str, arguments: dict) -> str:
        """
        Look up existing appointments for the caller by their phone number.

        Args:
            call_sid: Call identifier
            arguments: Function call arguments (not used, phone comes from context)

        Returns:
            Result message with appointment details or "no appointments found"
        """
        from services.calendar.ms_bookings_service import get_calendar_service

        context = self._conversations.get(call_sid)
        if not context or not context.phone_number:
            return "LOOKUP_ERROR: Cannot identify caller's phone number. Ask for their phone number or offer to transfer to a human."

        phone = context.phone_number
        calendar = get_calendar_service()

        try:
            appointments = await calendar.get_customer_appointments(phone)

            if not appointments:
                logger.info(f"Orchestrator: No appointments found for {phone[-4:]}")
                return "NO_APPOINTMENTS_FOUND: The caller has no upcoming appointments. Ask if they would like to book a new appointment."

            # Store appointments in context for later cancellation
            context.found_appointments = appointments

            # Format for LLM
            appt_list = []
            for i, appt in enumerate(appointments, 1):
                appt_list.append(
                    f"Appointment {i}: ID={appt['id']}, with {appt['staff_name']} on {appt['formatted_time']}"
                )

            appointments_text = "\n".join(appt_list)
            logger.info(f"Orchestrator: Found {len(appointments)} appointments for {phone[-4:]}")

            return f"APPOINTMENTS_FOUND: Found {len(appointments)} upcoming appointment(s):\n{appointments_text}\n\nTell the caller about their appointment(s) and ask if they want to cancel and reschedule. If they confirm cancellation, call cancel_booking with the appointment ID."

        except Exception as e:
            logger.error(f"Orchestrator: Error looking up appointments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"LOOKUP_ERROR: Could not look up appointments: {str(e)}. Offer to transfer to a human."

    async def _cancel_booking(self, call_sid: str, arguments: dict) -> str:
        """
        Cancel an existing appointment.

        Args:
            call_sid: Call identifier
            arguments: Must include confirm_cancel=True (appointment_id is taken from context)

        Returns:
            Result message
        """
        from services.calendar.ms_bookings_service import get_calendar_service
        from services.database import get_db_pool

        context = self._conversations.get(call_sid)
        confirm_cancel = arguments.get("confirm_cancel", False)

        if not confirm_cancel:
            return "CANCEL_ABORTED: Cancellation not confirmed. Ask if they still want to cancel or if they'd like to keep the appointment."

        # Get appointment ID from stored context (not from LLM arguments - LLM often hallucinates IDs)
        appointment_id = None
        if context and context.found_appointments and len(context.found_appointments) > 0:
            # Use the first found appointment (most common case: single appointment)
            appointment_id = context.found_appointments[0].get('id', '')
            logger.info(f"Orchestrator: Using appointment ID from context: {appointment_id[:30]}...")

        # Fallback to LLM argument only if context doesn't have it
        if not appointment_id:
            appointment_id = arguments.get("appointment_id", "")
            logger.warning(f"Orchestrator: No appointment in context, using LLM argument: {appointment_id}")

        if not appointment_id:
            return "CANCEL_ERROR: No appointment found. Call lookup_my_bookings first to find the appointment."

        calendar = get_calendar_service()

        try:
            success = await calendar.cancel_appointment(appointment_id)

            if success:
                logger.info(f"Orchestrator: Cancelled appointment {appointment_id}")

                # Update local database if we have the booking
                try:
                    pool = await get_db_pool()
                    await pool.execute(
                        "UPDATE bookings SET status = 'cancelled' WHERE ms_booking_id = $1",
                        appointment_id
                    )
                except Exception as db_err:
                    logger.warning(f"Orchestrator: Failed to update local booking status: {db_err}")

                # Send cancellation confirmation SMS
                if context and context.phone_number and context.found_appointments:
                    try:
                        from services.sms.telnyx_sms_service import get_sms_service
                        sms = get_sms_service()
                        if sms.is_available():
                            appt = context.found_appointments[0]
                            customer_name = context.caller_name or appt.get('customer_name', 'there')
                            staff_name = appt.get('staff_name', 'your accountant')
                            appt_time = appt.get('formatted_time', 'your scheduled time')
                            caller_lang = context.language or 'en'

                            import asyncio
                            asyncio.create_task(sms.send_cancellation_confirmation(
                                to_number=context.phone_number,
                                customer_name=customer_name,
                                staff_name=staff_name,
                                appointment_time=appt_time,
                                language=caller_lang,
                            ))
                            logger.info(f"Orchestrator: Cancellation SMS queued for {context.phone_number}")
                    except Exception as sms_err:
                        logger.warning(f"Orchestrator: Failed to send cancellation SMS: {sms_err}")

                # Clear found appointments from context
                context.found_appointments = None

                return "BOOKING_CANCELLED: The appointment has been cancelled successfully. Ask if they would like to book a new appointment at a different time."
            else:
                return "CANCEL_ERROR: Failed to cancel the appointment. Offer to transfer to a human for assistance."

        except Exception as e:
            logger.error(f"Orchestrator: Error cancelling appointment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"CANCEL_ERROR: System error while cancelling: {str(e)}. Offer to transfer to a human."

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

        # Add caller context to prompt
        if context.caller_name:
            system_prompt += f"\n\nCALLER CONTEXT:\n- The caller's name on file is: {context.caller_name}. Use their name naturally.\n- IMPORTANT: If the caller corrects their name or says their name is different (e.g. 'اسمي غادة مش غايد' / 'My name is Ghada not Gaid'), you MUST call register_caller_name with the CORRECT name immediately. The old name may be wrong or garbled from a previous call.\n- Treat the caller's own statement of their name as authoritative — always believe them over what's on file."
        else:
            system_prompt += "\n\nCALLER CONTEXT:\n- The caller's name is NOT yet known.\n- CRITICAL: As soon as the caller says their name, you MUST call the register_caller_name tool IMMEDIATELY. This is required before any booking can use their name.\n- If you already asked for their name and they reply with something that sounds like a name, call register_caller_name right away.\n- Also use their name in the booking (customer_name parameter) when they book an appointment."

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
            max_tokens=80,
            stream=False,
            tools=tools
        )

        try:
            llm_response = await self.llm.chat_with_tools(request)

            # Check if the LLM wants to call a function
            if llm_response.tool_calls:
                for tool_call in llm_response.tool_calls:
                    tool_name = tool_call["name"]
                    import json as _json
                    arguments = _json.loads(tool_call["arguments"])
                    logger.info(f"Orchestrator: LLM requested {tool_name}: {arguments}")

                    # --- Register caller name (LLM extracts it naturally) ---
                    if tool_name == "register_caller_name":
                        caller_name = arguments.get("caller_name", "").strip()
                        if caller_name and context:
                            context.caller_name = caller_name
                            context.name_collected = True
                            context.is_known_caller = True
                            await self.caller_service.register_caller(
                                context.phone_number,
                                caller_name,
                                context.language
                            )
                            logger.info(f"Orchestrator: LLM registered caller name: {caller_name}")

                        # Let the LLM generate a natural follow-up response
                        messages.append(Message(role=LLMRole.ASSISTANT, content=llm_response.content or ""))
                        messages.append(Message(role=LLMRole.USER, content=f"[SYSTEM: Caller name '{caller_name}' has been saved. Now greet them warmly by name and ask how you can help. Keep it brief.]"))

                        follow_up_request = LLMRequest(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=60,
                            stream=True
                        )
                        full_response = ""
                        async for chunk in self.llm.chat_stream(follow_up_request):
                            full_response += chunk.delta

                        response = full_response.strip()
                        context.add_assistant_message(response)
                        return response

                    # --- Transfer to human ---
                    if tool_name == "transfer_to_human":
                        reason = arguments.get("reason", "caller request")
                        logger.info(f"Orchestrator: LLM triggered transfer - reason: {reason}")
                        await self._handle_transfer(call_sid)
                        # _handle_transfer already spoke to the caller
                        # Return a marker so the outer code skips speaking
                        context.add_assistant_message("[transferred to human]")
                        return "__TRANSFER_HANDLED__"

                    # --- End call (LLM decided conversation is over) ---
                    if tool_name == "end_call":
                        reason = arguments.get("reason", "conversation complete")
                        logger.info(f"Orchestrator: LLM triggered end_call - reason: {reason}")

                        # If LLM also provided a goodbye message, speak it first
                        goodbye_text = (llm_response.content or "").strip()
                        if not goodbye_text:
                            # LLM returned tool call without text — ask it to generate farewell
                            lang_hint = "Arabic" if context.language == "ar" else "English"
                            caller_ref = f" The caller's name is {context.caller_name}." if context.caller_name else ""
                            messages.append(Message(role=LLMRole.ASSISTANT, content=""))
                            messages.append(Message(role=LLMRole.USER, content=(
                                f"[SYSTEM: The call is ending. Reason: {reason}.{caller_ref} "
                                f"Say a warm, brief goodbye in {lang_hint}. One sentence max.]"
                            )))
                            farewell_request = LLMRequest(
                                messages=messages,
                                temperature=0.7,
                                max_tokens=80,
                                stream=False
                            )
                            try:
                                farewell_response = await self.llm.chat(farewell_request)
                                goodbye_text = (farewell_response.content or "").strip().strip('"')
                            except Exception as e:
                                logger.warning(f"Orchestrator: Farewell LLM call failed: {e}")

                        if goodbye_text:
                            context.add_assistant_message(goodbye_text)
                            await self._speak_to_caller(call_sid, goodbye_text, context.language)

                        # End the call after goodbye is spoken
                        await self.end_call(call_sid)
                        return "__TRANSFER_HANDLED__"  # Reuse marker to skip outer speaking

                    # --- Booking tools ---
                    if tool_name in ("check_appointment", "confirm_appointment", "lookup_my_bookings", "cancel_booking"):
                        # Execute the appropriate booking step
                        if tool_name == "check_appointment":
                            booking_result = await self._check_booking(call_sid, arguments)
                        elif tool_name == "confirm_appointment":
                            booking_result = await self._confirm_booking(call_sid, arguments)
                        elif tool_name == "lookup_my_bookings":
                            booking_result = await self._lookup_my_bookings(call_sid, arguments)
                        elif tool_name == "cancel_booking":
                            booking_result = await self._cancel_booking(call_sid, arguments)

                        # Feed result back to LLM for a natural response
                        messages.append(Message(role=LLMRole.ASSISTANT, content=llm_response.content or ""))

                        if "BOOKING_NEEDS_RECHECK" in booking_result:
                            # LLM skipped re-checking after unavailable — tell caller we need to verify
                            messages.append(Message(role=LLMRole.USER, content=(
                                f"[SYSTEM: {booking_result}. "
                                f"Tell the caller: 'Let me check that time for you' or similar. "
                                f"Do NOT say there was an error. Keep it brief.]"
                            )))
                        else:
                            messages.append(Message(role=LLMRole.USER, content=f"[SYSTEM: {booking_result}. Reply in ONE short sentence (max 10 words). For unavailable: just say the time is taken and give ONE alternative. For confirmed: 'تم الحجز.' Do NOT list multiple times. Do NOT add 'anything else?'.]"))

                        # Get final response (streaming, no tools)
                        follow_up_request = LLMRequest(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=50,
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
                max_tokens=80,
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
        Speak response to caller using streaming TTS.

        KEY DESIGN: Audio is streamed to Twilio as it's generated by ElevenLabs.
        The caller starts hearing audio within ~200ms, not after 2-3s of synthesis.

        Pipeline: Text → ElevenLabs HTTP stream → raw ulaw chunks → Twilio WebSocket

        Args:
            call_sid: Call identifier
            text: Text to speak
            language: Language code
        """
        import traceback
        import time
        logger.info(f"Orchestrator: _speak_to_caller START - text: '{text[:50]}...'")

        context = self._conversations.get(call_sid)
        twilio_handler = self._twilio_handlers.get(call_sid)
        state_lock = self._call_state_locks.get(call_sid)

        if not context or not twilio_handler:
            logger.error(f"Orchestrator: Missing context or handler")
            return

        if not self.tts:
            logger.warning("TTS not available, cannot speak to caller")
            async with state_lock:
                context.state = ConversationState.LISTENING
            return

        try:
            # Clear the barge-in reset flag at the start of each TTS turn
            # This ensures we track fresh barge-ins for this utterance
            self._barge_in_reset_done[call_sid] = False

            # Pre-process Arabic text: convert numbers/times to Arabic words
            if language == "ar":
                from services.audio.arabic_text_processor import process_arabic_text
                text = process_arabic_text(text)

            from services.tts.tts_base import TTSRequest
            request = TTSRequest(
                text=text,
                language=language,
                voice_id=None
            )

            # =====================================================
            # STREAMING TTS → TWILIO PIPELINE
            # =====================================================
            # Stream audio chunks from ElevenLabs directly to Twilio
            # as they're generated. The caller hears audio within ~200ms.
            logger.info(f"Orchestrator: Starting streaming TTS to Twilio...")
            tts_start = time.time()

            audio_stream = self.tts.synthesize_stream_async(request)
            total_bytes = await twilio_handler.stream_audio_chunks(audio_stream)

            tts_elapsed = time.time() - tts_start
            logger.info(f"Orchestrator: TTS streaming complete - {total_bytes} bytes in {tts_elapsed:.2f}s")

            if total_bytes == 0:
                logger.error("Orchestrator: TTS streaming returned 0 bytes")
                return

            # =====================================================
            # POST-STREAM PLAYBACK WAIT
            # =====================================================
            # Audio was streamed in near-real-time, so most of it has
            # already been played by Twilio during the stream. We only
            # need to wait for the tail end + network buffer.
            #
            # With streaming: TTS generation time ≈ audio duration
            # (ElevenLabs generates roughly at real-time speed)
            # So by the time streaming ends, most audio has played.
            # We just wait a short buffer for the last chunks.
            audio_duration = total_bytes / 8000.0
            # Estimate how much audio is still buffered at Twilio
            # (audio_duration minus the time we spent streaming)
            remaining_playback = max(0, audio_duration - tts_elapsed)
            playback_buffer = min(remaining_playback + 0.2, audio_duration)
            step = 0.1

            if playback_buffer > 0:
                logger.info(f"Orchestrator: Waiting {playback_buffer:.1f}s for remaining playback (audio={audio_duration:.1f}s, streamed in {tts_elapsed:.1f}s)")
                elapsed = 0.0
                while elapsed < playback_buffer:
                    await asyncio.sleep(step)
                    elapsed += step

                    # Check for barge-in during remaining playback
                    if context.state != ConversationState.SPEAKING:
                        logger.info(f"Orchestrator: Barge-in during post-stream wait at {elapsed:.1f}s")
                        break

            # Reset STT for clean listening — reconnects to flush echo buffer
            # This gives STT a completely clean slate so the user's first words
            # are detected immediately instead of being lost in echo noise.
            # IMPORTANT: Skip if barge-in already triggered a reset (prevents double reset race condition)
            call_stt = self._call_stt_instances.get(call_sid)
            barge_in_already_reset = self._barge_in_reset_done.get(call_sid, False)

            if call_stt and not barge_in_already_reset:
                # LANGUAGE SAFETY NET: Detect the ACTUAL language of the LLM response
                # from its text content, and update STT if it differs.
                # This catches cases where the user requested a language switch
                # (e.g. "speak Arabic") but the keyword detection missed it —
                # the LLM understood and responded in Arabic, but STT stayed English.
                # We detect Arabic by character presence (definitive signal).
                import re
                response_has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))

                if response_has_arabic and call_stt.language != "ar":
                    logger.info(f"Orchestrator: LLM responded in Arabic but STT is '{call_stt.language}' — switching STT to Arabic before reset")
                    call_stt.language = "ar"
                    if context:
                        context.language = "ar"
                elif not response_has_arabic and call_stt.language == "ar":
                    # LLM responded in English (no Arabic chars) but STT is Arabic
                    logger.info(f"Orchestrator: LLM responded in English but STT is 'ar' — switching STT to English before reset")
                    call_stt.language = "en"
                    if context:
                        context.language = "en"

                await call_stt.reset_for_listening()
            elif barge_in_already_reset:
                logger.info(f"Orchestrator: Skipping post-TTS STT reset (barge-in already did it)")

            # Set echo guard for residual phone speaker echo (shortened since STT is clean)
            self._echo_guard_until[call_sid] = time.time() + 0.15
            logger.info(f"Orchestrator: Playback complete, echo guard 0.15s")

        except Exception as e:
            logger.error(f"Orchestrator: Exception in _speak_to_caller: {e}")
            logger.error(f"Orchestrator: Traceback:\n{traceback.format_exc()}")

        finally:
            async with state_lock:
                if context.state == ConversationState.SPEAKING:
                    context.state = ConversationState.LISTENING
            logger.info(f"Orchestrator: _speak_to_caller END (state={context.state.value})")


    async def _detect_barge_in(self, audio_data: bytes, call_sid: str = None) -> bool:
        """
        Detect if user is speaking during AI speech (barge-in)

        Uses energy-based detection with consecutive frame requirement
        to avoid false triggers from noise bursts.

        IMPORTANT: μ-law encoding maps silence to byte values 0xFF (255)
        and 0x7F (127). We must decode to linear PCM first to get real energy.

        Args:
            audio_data: Audio chunk to analyze (μ-law encoded)
            call_sid: Call identifier for per-call state

        Returns:
            True if user is speaking (barge-in detected)
        """
        import time

        sample_count = min(len(audio_data), 1000)
        if sample_count == 0:
            return False

        # Check grace period: skip barge-in detection for first 1.5s after speech starts
        # This prevents Twilio line noise from immediately killing playback
        sid = call_sid or "_default"
        now = time.time()
        grace_start = self._barge_in_speech_start.get(sid, 0)
        if grace_start > 0 and (now - grace_start) < 1.5:
            return False

        # Decode μ-law to linear PCM to get real energy values
        # μ-law lookup table for absolute linear value (simplified)
        # silence in μ-law = 0xFF/0x7F -> linear ~0
        # speech in μ-law deviates significantly
        import audioop
        try:
            linear_data = audioop.ulaw2lin(audio_data[:sample_count], 2)
            # Calculate RMS energy from 16-bit linear PCM
            rms = audioop.rms(linear_data, 2)
        except Exception:
            return False

        # Initialize per-call consecutive counter
        if sid not in self._barge_in_consecutive:
            self._barge_in_consecutive[sid] = 0

        # RMS threshold for actual speech over phone line (μ-law decoded)
        # Typical phone line noise: RMS ~14
        # Medium speech: RMS ~200
        # Loud speech: RMS ~900+
        # Use 200 as threshold to catch speech but ignore line noise
        threshold = 200

        if rms > threshold:
            self._barge_in_consecutive[sid] += 1
            # Require 5 consecutive high-energy frames (~100ms of speech) to trigger
            if self._barge_in_consecutive[sid] >= 5:
                logger.info(f"Orchestrator: Barge-in triggered - rms={rms}, threshold={threshold}, consecutive={self._barge_in_consecutive[sid]}")
                self._barge_in_consecutive[sid] = 0
                return True
        else:
            self._barge_in_consecutive[sid] = 0

        return False

    async def _reconnect_stt_with_language(self, call_sid: str, language: str, force: bool = False) -> None:
        """
        Reconnect the per-call STT instance with an explicit language code.
        This dramatically improves accuracy vs auto-detect on phone audio.

        Args:
            force: If True, reconnect even if STT already has a language set.
                   Used when we need to correct a wrong language choice.
        """
        call_stt = self._call_stt_instances.get(call_sid)
        if not call_stt:
            return

        # Skip if already set to the requested language
        if call_stt.language and call_stt.language == language:
            logger.info(f"Orchestrator: STT already set to '{language}', skipping reconnect")
            return

        # If not forcing, only reconnect from auto-detect mode
        if not force and call_stt.language and call_stt.language != "":
            logger.info(f"Orchestrator: STT already set to '{call_stt.language}', skipping reconnect (use force=True to override)")
            return

        try:
            logger.info(f"Orchestrator: Reconnecting STT with language='{language}' for call {call_sid}")
            if hasattr(call_stt, 'reconnect_with_language'):
                success = await call_stt.reconnect_with_language(language)
                if success:
                    logger.info(f"Orchestrator: STT reconnected with language='{language}'")
                else:
                    logger.error(f"Orchestrator: Failed to reconnect STT with language='{language}'")
            else:
                logger.warning(f"Orchestrator: STT provider does not support language reconnection")
        except Exception as e:
            logger.error(f"Orchestrator: Error reconnecting STT: {e}")

    async def _handle_transfer(self, call_sid: str) -> None:
        """
        Handle transfer to human by redirecting the Twilio call
        to the configured transfer phone number using TwiML <Dial>.
        """
        context = self._conversations.get(call_sid)
        settings = get_settings()
        transfer_number = settings.transfer_phone_number

        logger.info(f"Orchestrator: Call {call_sid} transferring to human, target={transfer_number}")

        if not transfer_number:
            logger.warning("Orchestrator: No TRANSFER_PHONE_NUMBER configured — cannot transfer")
            # Tell caller we can't transfer right now
            lang = context.language if context else "en"
            if lang == "ar":
                msg = "عذراً، لا يمكنني تحويلك حالياً. هل يمكنني مساعدتك بشيء آخر؟"
            else:
                msg = "I'm sorry, I'm unable to transfer you right now. Can I help you with something else?"
            await self._speak_to_caller(call_sid, msg, lang)
            return

        # Tell the caller we're transferring
        lang = context.language if context else "en"
        if lang == "ar":
            transfer_msg = "بالتأكيد، سأحولك الآن. لحظة من فضلك."
        else:
            transfer_msg = "Of course, let me transfer you now. One moment please."

        await self._speak_to_caller(call_sid, transfer_msg, lang)

        # Wait for the transfer message audio to finish playing
        # _speak_to_caller queues audio but it's sent asynchronously.
        # Estimate audio duration: μ-law 8kHz = 8000 bytes/sec.
        # A typical transfer message is ~3-4 seconds. Add buffer for safety.
        # We calculate from the last TTS response size if possible.
        await asyncio.sleep(5)  # 5 seconds is enough for transfer message + Twilio buffer

        # Use Twilio REST API to update the call with <Dial> TwiML
        try:
            import httpx

            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Calls/{call_sid}.json"

            # TwiML that dials the transfer number
            # Use a longer timeout (60s) and ringback to give the human time to answer
            if lang == "ar":
                fallback_say = "عذراً، الشخص الذي تحاول الوصول إليه غير متاح حالياً. يرجى المحاولة لاحقاً. مع السلامة."
                # Use Amazon Polly Zeina voice for natural Arabic TTS (much better than default Twilio)
                say_voice = "Polly.Zeina"
                say_lang = "ar-SA"
            else:
                fallback_say = "The person you are trying to reach is unavailable at the moment. Please try again later. Goodbye."
                say_voice = "Polly.Joanna"
                say_lang = "en-US"

            transfer_twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial callerId="{settings.twilio_phone_number}" timeout="60">
        <Number>{transfer_number}</Number>
    </Dial>
    <Say voice="{say_voice}" language="{say_lang}">{fallback_say}</Say>
</Response>'''

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    twilio_url,
                    auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                    data={"Twiml": transfer_twiml}
                )

                if response.status_code == 200:
                    logger.info(f"Orchestrator: Call {call_sid} successfully redirected to {transfer_number}")
                else:
                    logger.error(f"Orchestrator: Failed to redirect call: {response.status_code} - {response.text}")
                    if lang == "ar":
                        msg = "عذراً، حدث خطأ أثناء التحويل. هل يمكنني مساعدتك بشيء آخر؟"
                    else:
                        msg = "I'm sorry, there was an error transferring. Can I help you with something else?"
                    await self._speak_to_caller(call_sid, msg, lang)

        except Exception as e:
            logger.error(f"Orchestrator: Transfer exception: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def end_call(self, call_sid: str) -> None:
        """
        End a call and cleanup resources

        Args:
            call_sid: Call identifier
        """
        logger.info(f"Orchestrator: Ending call {call_sid}")

        # Capture context data before cleanup for database logging
        context = self._conversations.get(call_sid)
        caller_name = context.caller_name if context else None
        language = context.language if context else "en"
        turn_count = context.turn_count if context else 0
        booking_made = False
        transfer_requested = False

        # Check if booking was made during this call
        if context and context.pending_booking:
            booking_made = True

        # Check intent history for transfer requests
        if context and context.intent_history:
            for intent in context.intent_history:
                if intent == Intent.TRANSFER_TO_HUMAN:
                    transfer_requested = True
                    break

        # Update context
        if call_sid in self._conversations:
            self._conversations[call_sid].state = ConversationState.ENDED

        # =====================================================
        # Log call end to database
        # =====================================================
        try:
            from services.database import get_db_pool
            pool = await get_db_pool()
            # Update call log with end time and duration
            await pool.execute(
                """
                UPDATE call_logs
                SET ended_at = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER,
                    status = 'completed',
                    caller_name = COALESCE($2, caller_name),
                    language = COALESCE($3, language),
                    booking_made = $4,
                    transfer_requested = $5
                WHERE call_sid = $1
                """,
                call_sid,
                caller_name,
                language,
                booking_made,
                transfer_requested
            )
            logger.info(f"Orchestrator: Call end logged to database (duration calculated, booking={booking_made}, transfer={transfer_requested})")
        except Exception as db_err:
            logger.warning(f"Orchestrator: Failed to log call end to DB: {db_err}")

        # =====================================================
        # CRITICAL: Disconnect and cleanup this call's STT instance
        # =====================================================
        if call_sid in self._call_stt_instances:
            logger.info(f"Orchestrator: Disconnecting call-specific STT for {call_sid}")
            await self._call_stt_instances[call_sid].disconnect()
            del self._call_stt_instances[call_sid]

        # Clean up echo guard
        if call_sid in self._echo_guard_until:
            del self._echo_guard_until[call_sid]

        # Clean up garbled counter
        if call_sid in self._garbled_drop_count:
            del self._garbled_drop_count[call_sid]

        # Clean up barge-in per-call state
        if call_sid in self._barge_in_consecutive:
            del self._barge_in_consecutive[call_sid]
        if call_sid in self._barge_in_speech_start:
            del self._barge_in_speech_start[call_sid]
        if call_sid in self._barge_in_reset_done:
            del self._barge_in_reset_done[call_sid]

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

    def get_conversation_context(self, call_sid: str) -> Optional[ConversationContext]:
        """Get conversation context for a call"""
        return self._conversations.get(call_sid)

    def get_active_calls(self) -> list:
        """Get list of active calls with details for dashboard"""
        import time
        active_calls = []
        current_time = time.time()

        for call_sid, context in self._conversations.items():
            # Calculate duration in seconds
            duration = int(current_time - context.start_time)

            active_calls.append({
                "call_sid": call_sid,
                "phone_number": context.phone_number,
                "caller_name": context.caller_name or "Unknown",
                "language": context.language or "en",
                "state": context.state.value,
                "duration_seconds": duration,
                "turn_count": context.turn_count
            })

        return active_calls


# Global orchestrator instance
_orchestrator: Optional[ConversationOrchestrator] = None


def get_orchestrator() -> ConversationOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConversationOrchestrator()
    return _orchestrator
