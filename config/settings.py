"""
=====================================================
AI Voice Platform v2 - Configuration Module
=====================================================
Centralized configuration management using pydantic-settings
"""

import os
import json
from typing import List, Optional, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # =====================================================
    # APPLICATION
    # =====================================================
    app_name: str = "AI Voice Platform"
    app_version: str = "2.0.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ws_port: int = 8001
    # Public domain for external access (used for Twilio webhooks)
    public_domain: str = Field(default="", alias="PUBLIC_DOMAIN")
    # Store as string internally to avoid JSON parsing issues
    # Will be converted to List[str] by property
    allowed_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        alias="ALLOWED_ORIGINS"
    )

    # =====================================================
    # SECURITY
    # =====================================================
    secret_key: str = Field(..., alias="SECRET_KEY")
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    # WARNING: Only set to True in development with local Twilio testing
    # In production, always validate Twilio signatures
    skip_twilio_signature_validation: bool = Field(
        default=False,
        alias="SKIP_TWILIO_SIGNATURE_VALIDATION"
    )

    # =====================================================
    # DATABASE
    # =====================================================
    database_url: str = Field(..., alias="DATABASE_URL")
    postgres_user: str = Field(default="ai_voice", alias="POSTGRES_USER")
    postgres_password: str = Field(default="", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="ai_voice_db", alias="POSTGRES_DB")

    # =====================================================
    # REDIS
    # =====================================================
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_session_ttl: int = 3600  # 1 hour
    trusted_proxy_cidrs: str = Field(
        default="127.0.0.1/32,::1/128,172.16.0.0/12",
        alias="TRUSTED_PROXY_CIDRS",
    )

    # =====================================================
    # TWILIO
    # =====================================================
    twilio_account_sid: str = Field(..., alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(..., alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(..., alias="TWILIO_PHONE_NUMBER")

    # =====================================================
    # STT (Speech-to-Text) Provider
    # =====================================================
    stt_provider: str = Field(default="elevenlabs", alias="STT_PROVIDER")  # 'deepgram', 'whisper', or 'elevenlabs'

    # =====================================================
    # DEEPGRAM STT
    # =====================================================
    deepgram_api_key: str = Field(..., alias="DEEPGRAM_API_KEY")
    deepgram_model: str = "nova-2"  # nova-2 is the fastest and most accurate
    deepgram_language: str = "en-US"  # en-US, ar (Arabic), or mul (multilingual)
    deepgram_multilingual: bool = Field(default=False, alias="DEEPGRAM_MULTILINGUAL")  # Set True for Arabic/English support
    deepgram_smart_format: bool = True
    deepgram_paragraphs: bool = True
    deepgram_punctuate: bool = True
    deepgram_profanity_filter: bool = True
    deepgram_diairize: bool = False  # Not needed for single-caller scenarios

    # =====================================================
    # OPENAI WHISPER STT
    # =====================================================
    # Uses same openai_api_key from OPENAI section
    whisper_model: str = Field(default="whisper-1", alias="WHISPER_MODEL")
    whisper_language: str = Field(default="", alias="WHISPER_LANGUAGE")  # Empty = auto-detect, or 'en', 'ar', etc.
    whisper_silence_threshold: float = Field(default=0.3, alias="WHISPER_SILENCE_THRESHOLD")
    whisper_silence_duration: float = Field(default=1.0, alias="WHISPER_SILENCE_DURATION")  # Seconds of silence to trigger transcription
    whisper_min_audio_length: float = Field(default=0.5, alias="WHISPER_MIN_AUDIO_LENGTH")  # Minimum seconds before transcribing

    # =====================================================
    # ELEVENLABS TTS
    # =====================================================
    elevenlabs_api_key: str = Field(..., alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="Rachel", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.5  # 0-1, lower = more expressive
    elevenlabs_similarity_boost: float = 0.75  # 0-1, higher = more similar to original
    elevenlabs_output_format: str = "ulaw_8000"

    # =====================================================
    # ELEVENLABS STT (Scribe v2 Realtime)
    # =====================================================
    elevenlabs_stt_model: str = Field(default="scribe_v2_realtime", alias="ELEVENLABS_STT_MODEL")
    elevenlabs_stt_language: str = Field(default="", alias="ELEVENLABS_STT_LANGUAGE")  # Empty = auto-detect
    elevenlabs_stt_sample_rate: int = Field(default=8000, alias="ELEVENLABS_STT_SAMPLE_RATE")  # 8000 for Twilio ulaw
    elevenlabs_stt_filter_background_audio: bool = Field(
        default=False, alias="ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO"
    )
    barge_in_diagnostics_enabled: bool = Field(
        default=False, alias="BARGE_IN_DIAGNOSTICS_ENABLED"
    )
    speech_aware_barge_in_enabled: bool = Field(
        default=False, alias="SPEECH_AWARE_BARGE_IN_ENABLED"
    )
    local_vad_model_path: str = Field(
        default="models/vad/silero_vad.onnx", alias="LOCAL_VAD_MODEL_PATH"
    )
    local_vad_probability_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        alias="LOCAL_VAD_PROBABILITY_THRESHOLD",
    )
    local_vad_speech_duration_ms: int = Field(
        default=160, ge=32, le=320, alias="LOCAL_VAD_SPEECH_DURATION_MS"
    )
    local_vad_max_input_gap_ms: int = Field(
        default=96, ge=32, le=1000, alias="LOCAL_VAD_MAX_INPUT_GAP_MS"
    )
    local_vad_max_inference_ms: float = Field(
        default=20.0,
        gt=0.0,
        le=100.0,
        allow_inf_nan=False,
        alias="LOCAL_VAD_MAX_INFERENCE_MS",
    )

    @field_validator("local_vad_model_path")
    @classmethod
    def validate_local_vad_model_path(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 512 or "\x00" in value or not value.endswith(".onnx"):
            raise ValueError("local VAD model path must be a bounded ONNX path")
        return value

    # =====================================================
    # OPENAI
    # =====================================================
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000
    openai_stream: bool = True
    openai_presence_penalty: float = 0.0
    openai_frequency_penalty: float = 0.0

    # =====================================================
    # MICROSOFT GRAPH (Bookings)
    # =====================================================
    ms_bookings_tenant_id: str = Field(default="", alias="MS_BOOKINGS_TENANT_ID")
    ms_bookings_client_id: str = Field(default="", alias="MS_BOOKINGS_CLIENT_ID")
    ms_bookings_client_secret: str = Field(default="", alias="MS_BOOKINGS_CLIENT_SECRET")
    ms_bookings_business_id: str = Field(default="", alias="MS_BOOKINGS_BUSINESS_ID")
    booking_observation_enabled: bool = Field(
        default=False, alias="BOOKING_OBSERVATION_ENABLED"
    )
    booking_observation_interval_seconds: int = Field(
        default=60, ge=10, le=3600, alias="BOOKING_OBSERVATION_INTERVAL_SECONDS"
    )
    booking_observation_freshness_seconds: int = Field(
        default=180, ge=11, le=7200, alias="BOOKING_OBSERVATION_FRESHNESS_SECONDS"
    )
    automatic_notifications_enabled: bool = Field(
        default=False, alias="AUTOMATIC_NOTIFICATIONS_ENABLED"
    )
    verified_phone_booking_enabled: bool = Field(
        default=False, alias="VERIFIED_PHONE_BOOKING_ENABLED"
    )
    automatic_notification_workers_paused: bool = Field(
        default=False, alias="AUTOMATIC_NOTIFICATION_WORKERS_PAUSED"
    )
    automatic_notifications_interval_seconds: int = Field(
        default=60, ge=10, le=3600, alias="AUTOMATIC_NOTIFICATIONS_INTERVAL_SECONDS"
    )

    @model_validator(mode="after")
    def validate_observation_freshness(self):
        if (self.booking_observation_freshness_seconds
                <= self.booking_observation_interval_seconds):
            raise ValueError("observation freshness must exceed poll interval")
        if self.automatic_notifications_enabled and not self.booking_observation_enabled:
            raise ValueError("automatic notifications require provider observation")
        if (self.automatic_notification_workers_paused
                and not self.automatic_notifications_enabled):
            raise ValueError("paused notification workers require automatic notifications")
        return self

    # =====================================================
    # SMS (Telnyx)
    # =====================================================
    telnyx_api_key: str = Field(default="", alias="TELNYX_API_KEY")
    telnyx_phone_number: str = Field(default="", alias="TELNYX_PHONE_NUMBER")

    # =====================================================
    # CALL TRANSFER
    # =====================================================
    transfer_phone_number: str = Field(default="", alias="TRANSFER_PHONE_NUMBER")

    # =====================================================
    # FEATURES
    # =====================================================
    enable_analytics: bool = Field(default=True, alias="ENABLE_ANALYTICS")
    enable_recording: bool = Field(default=True, alias="ENABLE_RECORDING")
    max_call_duration: int = Field(default=600, alias="MAX_CALL_DURATION")
    default_language: str = Field(default="auto", alias="DEFAULT_LANGUAGE")

    # Conversation settings
    silence_timeout_seconds: int = 5  # How long to wait for user speech
    interruption_energy_threshold: float = 0.3  # Energy level for barge-in detection
    max_turns_without_intent: int = 3  # Max turns before escalating

    # =====================================================
    # MONITORING
    # =====================================================
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    # =====================================================
    # PROPERTIES
    # =====================================================
    @property
    def allowed_origins(self) -> List[str]:
        """Get allowed origins as a list"""
        return self._parse_origins_string(self.allowed_origins_str)

    def _parse_origins_string(self, origins_str: str) -> List[str]:
        """Parse origins from comma-separated string"""
        if not origins_str:
            return ["http://localhost:3000", "http://localhost:8000"]

        # Try JSON parsing first (for backward compatibility)
        try:
            parsed = json.loads(origins_str)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to comma-separated parsing
        origins = [origin.strip() for origin in origins_str.split(',')]
        return [o for o in origins if o]  # Filter out empty strings


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)"""
    return settings
