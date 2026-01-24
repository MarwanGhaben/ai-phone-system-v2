"""
=====================================================
AI Voice Platform v2 - Configuration Module
=====================================================
Centralized configuration management using pydantic-settings
"""

import os
from typing import List, Optional
from pydantic import Field
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
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="ALLOWED_ORIGINS"
    )

    # =====================================================
    # SECURITY
    # =====================================================
    secret_key: str = Field(..., alias="SECRET_KEY")
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

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

    # =====================================================
    # TWILIO
    # =====================================================
    twilio_account_sid: str = Field(..., alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(..., alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(..., alias="TWILIO_PHONE_NUMBER")

    # =====================================================
    # DEEPGRAM STT
    # =====================================================
    deepgram_api_key: str = Field(..., alias="DEEPGRAM_API_KEY")
    deepgram_model: str = "nova-2"  # nova-2 is the fastest and most accurate
    deepgram_language: str = "en-US"  # Will be overridden for auto-detect
    deepgram_smart_format: bool = True
    deepgram_paragraphs: bool = True
    deepgram_punctuate: bool = True
    deepgram_profanity_filter: bool = True
    deepgram_diairize: bool = False  # Not needed for single-caller scenarios

    # =====================================================
    # ELEVENLABS TTS
    # =====================================================
    elevenlabs_api_key: str = Field(..., alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="Rachel", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.5  # 0-1, lower = more expressive
    elevenlabs_similarity_boost: float = 0.75  # 0-1, higher = more similar to original
    elevenlabs_output_format: str = "mp3_44100_128"

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

    # =====================================================
    # SMS (Telnyx)
    # =====================================================
    telnyx_api_key: str = Field(default="", alias="TELNYX_API_KEY")
    telnyx_phone_number: str = Field(default="", alias="TELNYX_PHONE_NUMBER")

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


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)"""
    return settings
