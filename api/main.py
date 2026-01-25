"""
=====================================================
AI Voice Platform v2 - Main FastAPI Application
=====================================================
"""

import asyncio
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
    # Debug: Print all settings and headers
    print(f"DEBUG: public_domain from settings = '{getattr(settings, 'public_domain', 'NOT_FOUND')}'")
    print(f"DEBUG: host header = '{request.headers.get('host')}'")
    print(f"DEBUG: X-Forwarded-Host = '{request.headers.get('X-Forwarded-Host')}'")
    print(f"DEBUG: X-Forwarded-Proto = '{request.headers.get('X-Forwarded-Proto')}'")

    # Get WebSocket URL for this server
    # Use the domain from settings or fall back to host header
    domain = getattr(settings, 'public_domain', None)

    # Check if domain is set and not empty
    if domain and domain.strip():
        # Use configured public domain with secure WebSocket
        # Port 8443 bypasses Sophos SSL termination for WebSocket
        ws_url = f"wss://{domain.strip()}:8443/ws/calls"
        print(f"DEBUG: Using PUBLIC_DOMAIN: {ws_url}")
    else:
        # Try to get domain from X-Forwarded-Host header (set by reverse proxy)
        forwarded_host = request.headers.get('X-Forwarded-Host', '')
        if forwarded_host:
            # Use the forwarded host (public domain)
            ws_url = f"wss://{forwarded_host}/ws/calls"
            print(f"DEBUG: Using X-Forwarded-Host: {ws_url}")
        else:
            # Last resort: use host header but warn
            host = request.headers.get('host', 'localhost:8000')
            print(f"WARNING: No PUBLIC_DOMAIN set, using host header: {host}")
            # Convert to WebSocket URL - use standard port 443 (no port in URL)
            if host.startswith('http:'):
                host = host.replace('http:', 'https:')
            elif not host.startswith(('https:', 'wss:')):
                proto = request.headers.get('X-Forwarded-Proto', 'https')
                host = f"{proto}://{host}"
            # Remove port if present (use standard 443)
            if ':443' in host:
                host = host.split(':443')[0]
            elif ':80' in host:
                host = host.split(':80')[0]
            elif ':' in host and not (host.count(':') > 1):  # IPv6 has multiple colons
                host = host.split(':')[0]
            ws_url = host.replace('https://', 'wss://').replace('http://', 'ws://')
            if not ws_url.endswith('/ws/calls'):
                ws_url = f"{ws_url}/ws/calls"
            print(f"DEBUG: Fallback WebSocket URL: {ws_url}")

    from services.telephony.twilio_service import TwilioService
    twilio = TwilioService(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        phone_number=settings.twilio_phone_number
    )

    twiml = await twilio.generate_twiml(ws_url)

    # Log the WebSocket URL being sent to Twilio
    print(f"DEBUG: FINAL WebSocket URL: {ws_url}")
    print(f"DEBUG: Full TwiML:\n{twiml}")

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

    Note: Twilio sends CallSid in the 'start' event payload, not query string.
    """
    # Accept WebSocket connection
    await websocket.accept()

    # Get orchestrator
    orchestrator = get_orchestrator()

    import json

    # Read the first message to get call info, then pass to orchestrator
    # We only need to peek at the start event to extract call_sid and stream_sid
    call_sid = None
    phone_number = None
    stream_sid = ""

    try:
        # Peek at messages until we get the start event
        while call_sid is None:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                start_data = data.get("start", {})
                call_sid = start_data.get("callSid", "unknown")
                stream_sid = start_data.get("streamSid", "")
                # Check if phone number is in custom parameters
                phone_number = start_data.get("customParameters", {}).get("callerNumber")
                logger.info(f"WebSocket: Got start event for call {call_sid}, stream {stream_sid}")
                break
            elif event == "connected":
                logger.debug(f"WebSocket: Connected event received")
                continue
            elif event == "disconnect":
                logger.info("WebSocket: Disconnected before start event")
                return

        # Use default phone number if not provided
        if not phone_number:
            phone_number = "+0000000000"
            logger.warning(f"WebSocket: No phone number in custom parameters, using default")

        # Pass control to orchestrator - it will handle the rest
        logger.info(f"WebSocket: Passing control to orchestrator for call {call_sid}")
        await orchestrator.handle_call(call_sid, phone_number, websocket, stream_sid)

    except WebSocketDisconnect:
        logger.info(f"WebSocket: Disconnected")
    except Exception as e:
        logger.error(f"WebSocket: Error: {e}")
    finally:
        logger.info(f"WebSocket: Connection ended")


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


@app.get("/api/callers/recent")
async def get_recent_callers(limit: int = 10):
    """Get recent callers"""
    from services.callers.caller_service import get_caller_service
    caller_service = get_caller_service()
    return {"callers": caller_service.get_recent_callers(limit)}


@app.get("/api/callers/frequent")
async def get_frequent_callers(limit: int = 5, min_calls: int = 2):
    """Get frequent callers (VIPs)"""
    from services.callers.caller_service import get_caller_service
    caller_service = get_caller_service()
    return {"callers": caller_service.get_frequent_callers(limit, min_calls)}


@app.get("/api/callers/stats")
async def get_caller_stats():
    """Get caller statistics"""
    from services.callers.caller_service import get_caller_service
    caller_service = get_caller_service()
    all_callers = caller_service.get_all_callers()

    total_unique = len(all_callers)
    returning = sum(1 for c in all_callers.values() if c.get("call_count", 0) > 1)

    return {
        "total_unique": total_unique,
        "returning": returning,
        "new": total_unique - returning
    }


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
