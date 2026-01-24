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
# ADMIN API
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
# DASHBOARD API
# =====================================================

from services.dashboard.dashboard_service import get_dashboard_service

@app.get("/api/dashboard/operational")
async def get_operational_overview():
    """Get operational overview for dashboard"""
    dashboard = get_dashboard_service()
    return await dashboard.get_operational_overview()


@app.get("/api/dashboard/calls")
async def get_dashboard_call_analytics():
    """Get call analytics for dashboard"""
    dashboard = get_dashboard_service()
    return await dashboard.get_call_analytics()


@app.get("/api/dashboard/api-usage")
async def get_dashboard_api_monitoring():
    """Get API usage monitoring for dashboard"""
    dashboard = get_dashboard_service()
    return await dashboard.get_api_monitoring()


@app.get("/api/dashboard/health")
async def get_dashboard_system_health():
    """Get system health for dashboard"""
    dashboard = get_dashboard_service()
    return await dashboard.get_system_health()


@app.get("/api/dashboard/bookings")
async def get_dashboard_bookings():
    """Get bookings overview for dashboard"""
    dashboard = get_dashboard_service()
    return await dashboard.get_bookings_overview()


# =====================================================
# KNOWLEDGE BASE API
# =====================================================

from services.knowledge.faq_service import get_kb_service

@app.get("/api/accountants")
async def get_accountants():
    """Get all accountants from configuration"""
    from services.config.accountants_service import get_accountants_service
    acc_service = get_accountants_service()
    return {"accountants": acc_service.get_all_accountants()}


@app.get("/api/kb/categories")
async def get_kb_categories():
    """Get all FAQ categories"""
    kb = get_kb_service()
    return {"categories": kb.get_all_categories()}


@app.get("/api/kb/faqs")
async def get_kb_faqs(language: str = "en", category: str = None):
    """Get all FAQs, optionally filtered by category"""
    kb = get_kb_service()

    if category:
        faqs = kb.get_faqs_by_category(category, language)
    else:
        faqs = kb.get_all_faqs(language)

    return {"faqs": faqs, "language": language}


# =====================================================
# CALENDAR API
# =====================================================

from services.calendar.ms_bookings_service import get_calendar_service

@app.get("/api/calendar/staff")
async def get_calendar_staff():
    """Get all staff members from calendar"""
    calendar = get_calendar_service()
    staff = await calendar.get_staff_members()
    return {
        "staff": [
            {"id": s.id, "name": s.name, "email": s.email, "role": s.role}
            for s in staff
        ]
    }


@app.get("/api/calendar/services")
async def get_calendar_services():
    """Get all services from calendar"""
    calendar = get_calendar_service()
    services = await calendar.get_services()
    return {
        "services": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "duration_minutes": s.duration_minutes,
                "price": s.price
            }
            for s in services
        ]
    }


@app.get("/api/calendar/availability")
async def get_calendar_availability(service_id: str, staff_id: str = None, days_ahead: int = 7):
    """Get available time slots"""
    calendar = get_calendar_service()
    slots = await calendar.get_available_slots(service_id, staff_id, days_ahead)
    return {
        "slots": [
            {
                "staff_id": s.staff_id,
                "staff_name": s.staff_name,
                "start_time": s.start_time.isoformat(),
                "end_time": s.end_time.isoformat(),
                "formatted": s.formatted
            }
            for s in slots
        ]
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
