# AI Voice Platform v2 - Enterprise Edition

**Next-generation AI voice agent platform** for handling phone calls with human-level conversation quality.

## Architecture

```
Twilio Media Streams → Orchestrator → Deepgram (STT) → GPT-4o (LLM) → ElevenLabs (TTS) → Caller
                                    ↓
                              Redis State
                              PostgreSQL Log
```

## Features

- ✅ **Real-time streaming** - <500ms end-to-end latency
- ✅ **Auto language detection** - English/Arabic without menus
- ✅ **True barge-in** - Interrupt AI anytime by speaking
- ✅ **Human-quality voice** - ElevenLabs TTS
- ✅ **Enterprise STT** - Deepgram Nova-2
- ✅ **GPT-4o reasoning** - Best-in-class AI
- ✅ **Multi-tenant** - Serve multiple businesses
- ✅ **Docker deployment** - One-command setup

## Quick Start

### 1. Clone & Configure

```bash
cd ai-voice-platform-v2
cp .env.example .env
# Edit .env with your API keys
```

### 2. Deploy (Docker Compose)

```bash
docker-compose up -d
```

That's it! The platform will start on port 8000.

### 3. Configure Twilio

Point your Twilio number to:
```
https://your-server.com/api/incoming-call
```

## Environment Variables

Required in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `TWILIO_ACCOUNT_SID` | Twilio account | AC... |
| `TWILIO_AUTH_TOKEN` | Twilio token | ... |
| `DEEPGRAM_API_KEY` | Deepgram STT | ... |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS | ... |
| `OPENAI_API_KEY` | OpenAI LLM | sk-... |
| `SECRET_KEY` | Security key | random string |

## Project Structure

```
ai-voice-platform-v2/
├── api/                    # FastAPI application
│   └── main.py            # Main app with WebSocket endpoint
├── services/
│   ├── telephony/          # Twilio integration
│   ├── stt/                # Speech-to-Text (Deepgram)
│   ├── tts/                # Text-to-Speech (ElevenLabs)
│   ├── llm/                # LLM (OpenAI)
│   ├── conversation/       # Orchestrator
│   └── analytics/          # Metrics & logging
├── models/                 # Database models
├── config/                 # Settings
├── scripts/                # Setup scripts
├── logs/                   # Application logs
├── Dockerfile              # Container image
├── docker-compose.yml      # Full stack deployment
└── requirements.txt        # Python dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/incoming-call` | GET | Twilio webhook (returns TwiML) |
| `/ws/calls` | WS | Twilio Media Streams |
| `/api/stats` | GET | Platform statistics |
| `/api/calls` | GET | Recent calls |

## Cost Estimate

For 100 calls/day, 3 min average, tax season (3 months):

| Service | Monthly |
|---------|---------|
| Twilio Media Streams | ~$86 |
| Deepgram STT | ~$59 |
| ElevenLabs TTS | ~$50 |
| GPT-4o | ~$11 |
| Hosting | ~$40 |
| **Total** | **~$250/month** |

**vs Human receptionist: $3,000-4,000/month**

## Development

Run locally without Docker:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.yml --profile production up -d

# View logs
docker-compose logs -f app

# Scale workers
docker-compose up -d --scale app=4
```

## Monitoring

- Logs: `./logs/ai-voice-platform.log`
- Health: `http://localhost:8000/health`
- Stats: `http://localhost:8000/api/stats`

## License

Proprietary - All rights reserved

---

**Built by Enterprise AI Architects**
