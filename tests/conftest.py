import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ.update(
    {
        "SECRET_KEY": "test-secret-key",
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": "test-token",
        "TWILIO_PHONE_NUMBER": "+14165550100",
        "DEEPGRAM_API_KEY": "test-deepgram-key",
        "ELEVENLABS_API_KEY": "test-elevenlabs-key",
        "OPENAI_API_KEY": "test-openai-key",
    }
)
