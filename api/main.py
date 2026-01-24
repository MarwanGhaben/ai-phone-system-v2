"""
=====================================================
AI Voice Platform v2 - Main FastAPI Application
=====================================================
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import get_settings
from services.conversation.orchestrator import get_orchestrator


# Get settings
settings = get_settings()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "../logs/ai-voice-platform.log",
    rotation="500 MB",
    level=settings.log_level,
    backtrace=True,
    diagnose=True
)
logger.add(lambda msg: print(msg, end=""), level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("AI Voice Platform v2 starting up...")
    yield
    logger.info("AI Voice Platform v2 shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Voice Platform v2",
    description="Enterprise-grade AI voice agent platform",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-voice-platform",
        "version": "2.0.0",
        "environment": settings.environment
    }


# =====================================================
# TWILIO WEBHOOK (TwiML)
# =====================================================

@app.get("/api/incoming-call")
async def incoming_call(request: Request):
    """
    Handle incoming call from Twilio
    Returns TwiML to connect to Media Stream
    """
    # Get WebSocket URL for this server
    ws_url = f"ws://{request.headers.get('host', 'localhost:8001')}/ws/calls"

    from services.telephony.twilio_service import TwilioService
    twilio = TwilioService(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        phone_number=settings.twilio_phone_number
    )

    twiml = await twilio.generate_twiml(ws_url)

    return HTMLResponse(content=twiml, media_type="application/xml")


# =====================================================
# WEBSOCKET ENDPOINT (Twilio Media Streams)
# =====================================================

@app.websocket("/ws/calls")
async def websocket_call_handler(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams

    This is where the real-time audio streaming happens.
    Twilio connects here when a call comes in.
    """
    # Accept WebSocket connection
    await websocket.accept()

    # Extract call parameters from query string
    call_sid = websocket.query_params.get("CallSid", "")
    phone_number = websocket.query_params.get("From", "").replace(":", "")

    if not call_sid:
        logger.warning("WebSocket: Missing CallSid, closing connection")
        await websocket.close()
        return

    logger.info(f"WebSocket: New call {call_sid} from {phone_number}")

    # Get orchestrator and handle the call
    orchestrator = get_orchestrator()

    # Import Twilio handler
    from services.conversation.orchestrator import TwilioMediaStreamHandler

    # Create handler and process
    handler = TwilioMediaStreamHandler(call_sid, websocket)
    await handler.handle_connection()

    logger.info(f"WebSocket: Call {call_sid} ended")


# =====================================================
# ADMIN API (Basic)
# =====================================================

@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    return {
        "active_calls": len(get_orchestrator()._conversations),
        "environment": settings.environment,
        "version": "2.0.0"
    }


@app.get("/api/calls")
async def list_calls():
    """List recent calls (TODO: implement database query)"""
    return {
        "calls": []  # TODO: Query from database
    }


# =====================================================
# ERROR HANDLERS
# =====================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# =====================================================
# MAIN ENTRY POINT (for development)
# =====================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
