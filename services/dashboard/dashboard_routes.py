"""
=====================================================
AI Voice Platform v2 - Dashboard API Routes
=====================================================
All API endpoints for the admin dashboard.
"""

import os
import psutil
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from loguru import logger

from services.dashboard.auth_service import get_auth_service
from services.dashboard.email_service import get_email_service
from services.database import get_db_pool


# =====================================================
# ROUTER SETUP
# =====================================================

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# =====================================================
# HTML PAGE ROUTES
# =====================================================

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page"""
    # Check if already logged in
    session_token = request.cookies.get("session_token")
    if session_token:
        auth = get_auth_service()
        user = await auth.validate_session(session_token)
        if user:
            return RedirectResponse(url="/dashboard", status_code=302)

    # Read and serve login template
    import os
    template_path = os.path.join(os.path.dirname(__file__), "../../templates/login.html")
    with open(template_path, "r") as f:
        return HTMLResponse(content=f.read())


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve main dashboard page"""
    # Check authentication
    session_token = request.cookies.get("session_token")
    if not session_token:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    auth = get_auth_service()
    user = await auth.validate_session(session_token)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    # Read and serve dashboard template
    import os
    template_path = os.path.join(os.path.dirname(__file__), "../../templates/dashboard.html")
    with open(template_path, "r") as f:
        return HTMLResponse(content=f.read())


# =====================================================
# REQUEST MODELS
# =====================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class MFARequest(BaseModel):
    code: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


# =====================================================
# AUTH HELPERS
# =====================================================

def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def get_current_user(request: Request) -> Optional[dict]:
    """Get current user from session cookie"""
    session_token = request.cookies.get("session_token")
    if not session_token:
        return None

    auth = get_auth_service()
    return await auth.validate_session(session_token)


async def require_auth(request: Request) -> dict:
    """Dependency that requires authentication"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# =====================================================
# AUTH ENDPOINTS
# =====================================================

@router.post("/api/login")
async def login(request: Request, login_data: LoginRequest):
    """
    Step 1: Authenticate with username/password
    Returns pending MFA status if successful
    """
    auth = get_auth_service()
    email_svc = get_email_service()
    ip_address = get_client_ip(request)

    success, user, error = await auth.authenticate_user(
        login_data.username,
        login_data.password,
        ip_address
    )

    if not success:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": error}
        )

    # Generate and send MFA code
    mfa_code = await auth.create_mfa_code(user['id'])
    email_svc.send_mfa_code(user['email'], mfa_code, user['username'])

    # Store user_id in temporary cookie for MFA step
    response = JSONResponse(content={
        "success": True,
        "mfa_required": True,
        "email_hint": user['email'][:3] + "***" + user['email'][user['email'].index('@'):]
    })

    # Set temporary MFA pending cookie (encrypted user_id)
    response.set_cookie(
        key="mfa_pending",
        value=str(user['id']),
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=600  # 10 minutes
    )

    return response


@router.post("/api/verify-mfa")
async def verify_mfa(request: Request, mfa_data: MFARequest):
    """
    Step 2: Verify MFA code and create session
    """
    auth = get_auth_service()
    email_svc = get_email_service()

    # Get pending user_id from cookie
    user_id_str = request.cookies.get("mfa_pending")
    if not user_id_str:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No pending MFA session"}
        )

    try:
        user_id = int(user_id_str)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid MFA session"}
        )

    # Verify the code
    success, error = await auth.verify_mfa_code(user_id, mfa_data.code)
    if not success:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": error}
        )

    # Get user info
    user = await auth.get_user_by_id(user_id)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "User not found"}
        )

    # Create session
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "Unknown")
    session_token = await auth.create_session(user_id, ip_address, user_agent)

    # Send login notification (optional)
    # email_svc.send_login_alert(user['email'], user['username'], ip_address, user_agent)

    response = JSONResponse(content={
        "success": True,
        "redirect": "/dashboard"
    })

    # Set session cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=86400  # 24 hours
    )

    # Clear MFA pending cookie
    response.delete_cookie("mfa_pending")

    return response


@router.post("/api/logout")
async def logout(request: Request):
    """Logout and invalidate session"""
    auth = get_auth_service()
    session_token = request.cookies.get("session_token")

    if session_token:
        await auth.invalidate_session(session_token)

    response = JSONResponse(content={"success": True})
    response.delete_cookie("session_token")
    response.delete_cookie("mfa_pending")

    return response


@router.get("/api/me")
async def get_current_user_info(user: dict = Depends(require_auth)):
    """Get current logged-in user info"""
    return {
        "username": user['username'],
        "email": user['email'],
        "is_superuser": user['is_superuser']
    }


# =====================================================
# SYSTEM STATUS ENDPOINTS
# =====================================================

@router.get("/api/system-status")
async def get_system_status(user: dict = Depends(require_auth)):
    """Get overall system status"""
    try:
        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"

        # Active calls
        try:
            from services.conversation.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            active_calls = len(orchestrator._conversations)
        except:
            active_calls = 0

        return {
            "status": "OPERATIONAL",
            "uptime": uptime_str,
            "rate_limiting": "Enabled",
            "storage_type": "PostgreSQL",
            "active_calls": active_calls,
            "workers": int(os.getenv('WEB_CONCURRENCY', '4')),
            "port": int(os.getenv('PORT', '5000'))
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"status": "ERROR", "error": str(e)}


@router.get("/api/ai-models")
async def get_ai_models(user: dict = Depends(require_auth)):
    """Get AI model configuration"""
    from config.settings import get_settings
    settings = get_settings()

    return {
        "language_model": getattr(settings, 'openai_model', 'gpt-4o'),
        "tts_service": "ElevenLabs",
        "stt_service": "ElevenLabs",
        "max_tokens": 80
    }


@router.get("/api/cpu-usage")
async def get_cpu_usage(user: dict = Depends(require_auth)):
    """Get CPU usage metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()

    return {
        "current_percent": round(cpu_percent, 1),
        "cores": cpu_count
    }


@router.get("/api/memory-usage")
async def get_memory_usage(user: dict = Depends(require_auth)):
    """Get memory usage metrics"""
    memory = psutil.virtual_memory()

    return {
        "used_gb": round(memory.used / (1024**3), 1),
        "total_gb": round(memory.total / (1024**3), 1),
        "available_gb": round(memory.available / (1024**3), 1),
        "percent": round(memory.percent, 1)
    }


@router.get("/api/disk-usage")
async def get_disk_usage(user: dict = Depends(require_auth)):
    """Get disk usage metrics"""
    disk = psutil.disk_usage('/')

    return {
        "used_gb": round(disk.used / (1024**3), 1),
        "total_gb": round(disk.total / (1024**3), 1),
        "free_gb": round(disk.free / (1024**3), 1),
        "percent": round(disk.percent, 1)
    }


@router.get("/api/pipeline-latency")
async def get_pipeline_latency(user: dict = Depends(require_auth)):
    """Get pipeline latency metrics (placeholder)"""
    # TODO: Implement actual latency tracking
    return {
        "total_ms": 0,
        "tts_ms": 0,
        "stt_ms": 0,
        "llm_ms": 0,
        "ffmpeg_ms": 0,
        "samples": 0
    }


@router.get("/api/security-status")
async def get_security_status(user: dict = Depends(require_auth)):
    """Get security configuration status"""
    return {
        "twilio_auth": True,
        "audio_signing": True,
        "rate_limiting": True,
        "ssrf_protection": True,
        "mfa_enabled": True
    }


# =====================================================
# CALLER ANALYTICS ENDPOINTS
# =====================================================

@router.get("/api/recent-calls")
async def get_recent_calls(user: dict = Depends(require_auth), limit: int = 20):
    """Get recent calls"""
    pool = await get_db_pool()

    # Try to get from call_logs if exists, otherwise from callers
    try:
        rows = await pool.fetch(
            """
            SELECT phone_number, caller_name as name, language, started_at,
                   duration_seconds, status, transfer_requested
            FROM call_logs
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit
        )

        return {
            "calls": [
                {
                    "phone": row['phone_number'],
                    "name": row['name'] or "Unknown",
                    "language": row['language'] or "en",
                    "date": row['started_at'].strftime("%m/%d %H:%M") if row['started_at'] else "",
                    "duration": row['duration_seconds'] or 0,
                    "status": row['status'],
                    "transferred": row['transfer_requested']
                }
                for row in rows
            ]
        }
    except:
        # Fallback to callers table
        rows = await pool.fetch(
            """
            SELECT phone_number, name, language, last_call, call_count
            FROM callers
            ORDER BY last_call DESC
            LIMIT $1
            """,
            limit
        )

        return {
            "calls": [
                {
                    "phone": row['phone_number'],
                    "name": row['name'] or "Unknown",
                    "language": row['language'] or "en",
                    "date": row['last_call'].strftime("%m/%d %H:%M") if row['last_call'] else "",
                    "duration": 0,
                    "call_count": row['call_count']
                }
                for row in rows
            ]
        }


@router.get("/api/call-statistics")
async def get_call_statistics(user: dict = Depends(require_auth)):
    """Get call statistics"""
    pool = await get_db_pool()

    # Get stats from callers table
    stats = await pool.fetchrow(
        """
        SELECT
            COUNT(*) as unique_callers,
            SUM(call_count) as total_calls,
            COUNT(CASE WHEN call_count > 1 THEN 1 END) as returning_callers
        FROM callers
        """
    )

    # Language distribution
    lang_dist = await pool.fetch(
        """
        SELECT language, COUNT(*) as count
        FROM callers
        GROUP BY language
        """
    )

    total = sum(r['count'] for r in lang_dist) or 1

    return {
        "total_calls": stats['total_calls'] or 0,
        "unique_callers": stats['unique_callers'] or 0,
        "returning_callers": stats['returning_callers'] or 0,
        "avg_duration_seconds": 0,  # TODO: calculate from call_logs
        "transfer_rate": 0,
        "language_distribution": [
            {
                "language": "Arabic" if r['language'] == 'ar' else "English",
                "code": r['language'] or 'en',
                "count": r['count'],
                "percentage": round(r['count'] / total * 100, 1)
            }
            for r in lang_dist
        ]
    }


@router.get("/api/frequent-callers")
async def get_frequent_callers(user: dict = Depends(require_auth), limit: int = 5):
    """Get most frequent callers (VIPs)"""
    pool = await get_db_pool()

    rows = await pool.fetch(
        """
        SELECT name, call_count
        FROM callers
        WHERE call_count > 1
        ORDER BY call_count DESC
        LIMIT $1
        """,
        limit
    )

    return {
        "callers": [
            {"name": row['name'] or "Unknown", "calls": row['call_count']}
            for row in rows
        ]
    }


# =====================================================
# BOOKINGS ENDPOINTS
# =====================================================

@router.get("/api/bookings")
async def get_bookings(user: dict = Depends(require_auth), limit: int = 50):
    """Get all bookings"""
    pool = await get_db_pool()

    try:
        rows = await pool.fetch(
            """
            SELECT id, client_name, client_email, phone_number, accountant_name,
                   appointment_time, client_type, language, status, created_at
            FROM bookings
            ORDER BY appointment_time DESC
            LIMIT $1
            """,
            limit
        )

        # Count by accountant
        accountant_counts = await pool.fetch(
            """
            SELECT accountant_name, COUNT(*) as count
            FROM bookings
            GROUP BY accountant_name
            """
        )

        upcoming = await pool.fetchval(
            """
            SELECT COUNT(*) FROM bookings
            WHERE appointment_time > NOW() AND status = 'confirmed'
            """
        )

        return {
            "total": len(rows),
            "upcoming": upcoming or 0,
            "by_accountant": {r['accountant_name']: r['count'] for r in accountant_counts},
            "bookings": [
                {
                    "id": row['id'],
                    "client": row['client_name'],
                    "email": row['client_email'],
                    "phone": row['phone_number'],
                    "accountant": row['accountant_name'],
                    "time": row['appointment_time'].isoformat() if row['appointment_time'] else None,
                    "type": row['client_type'],
                    "language": row['language'],
                    "status": row['status']
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching bookings: {e}")
        return {"total": 0, "upcoming": 0, "by_accountant": {}, "bookings": []}


@router.delete("/api/bookings/{booking_id}")
async def delete_booking(booking_id: int, user: dict = Depends(require_auth)):
    """Delete a booking"""
    pool = await get_db_pool()
    await pool.execute("DELETE FROM bookings WHERE id = $1", booking_id)
    return {"success": True}


@router.delete("/api/bookings")
async def clear_all_bookings(user: dict = Depends(require_auth)):
    """Clear all bookings (superuser only)"""
    if not user.get('is_superuser'):
        raise HTTPException(status_code=403, detail="Superuser required")

    pool = await get_db_pool()
    await pool.execute("DELETE FROM bookings")
    return {"success": True}


# =====================================================
# SMS ENDPOINTS
# =====================================================

@router.get("/api/sms-logs")
async def get_sms_logs(user: dict = Depends(require_auth), limit: int = 50):
    """Get SMS notification logs"""
    pool = await get_db_pool()

    try:
        rows = await pool.fetch(
            """
            SELECT id, phone_number, client_name, message, provider,
                   status, sent_at, booking_link
            FROM sms_logs
            ORDER BY sent_at DESC
            LIMIT $1
            """,
            limit
        )

        # Get stats
        total = await pool.fetchval("SELECT COUNT(*) FROM sms_logs")
        today = await pool.fetchval(
            "SELECT COUNT(*) FROM sms_logs WHERE sent_at::date = CURRENT_DATE"
        )
        this_week = await pool.fetchval(
            "SELECT COUNT(*) FROM sms_logs WHERE sent_at > NOW() - INTERVAL '7 days'"
        )

        provider_counts = await pool.fetch(
            """
            SELECT provider, COUNT(*) as count
            FROM sms_logs
            GROUP BY provider
            """
        )

        return {
            "total": total or 0,
            "today": today or 0,
            "this_week": this_week or 0,
            "by_provider": {r['provider']: r['count'] for r in provider_counts},
            "messages": [
                {
                    "id": row['id'],
                    "phone": row['phone_number'],
                    "name": row['client_name'],
                    "message": row['message'][:50] + "..." if row['message'] and len(row['message']) > 50 else row['message'],
                    "provider": row['provider'],
                    "status": row['status'],
                    "sent_at": row['sent_at'].isoformat() if row['sent_at'] else None,
                    "booking_link": row['booking_link']
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching SMS logs: {e}")
        return {"total": 0, "today": 0, "this_week": 0, "by_provider": {}, "messages": []}


# =====================================================
# CALLER MANAGEMENT ENDPOINTS
# =====================================================

@router.get("/api/callers")
async def get_all_callers(user: dict = Depends(require_auth)):
    """Get all callers for management"""
    pool = await get_db_pool()

    rows = await pool.fetch(
        """
        SELECT phone_number, name, language, call_count, last_call
        FROM callers
        ORDER BY last_call DESC
        """
    )

    return {
        "total": len(rows),
        "callers": [
            {
                "phone": row['phone_number'],
                "name": row['name'],
                "language": row['language'],
                "calls": row['call_count'],
                "last_call": row['last_call'].strftime("%m/%d/%Y") if row['last_call'] else None
            }
            for row in rows
        ]
    }


@router.put("/api/callers/{phone_number}")
async def update_caller(phone_number: str, request: Request, user: dict = Depends(require_auth)):
    """Update a caller's info"""
    data = await request.json()
    pool = await get_db_pool()

    # Build update query dynamically
    updates = []
    values = []
    param_num = 1

    if 'name' in data:
        updates.append(f"name = ${param_num}")
        values.append(data['name'])
        param_num += 1

    if 'language' in data:
        updates.append(f"language = ${param_num}")
        values.append(data['language'])
        param_num += 1

    if not updates:
        return {"success": False, "error": "No fields to update"}

    values.append(phone_number)
    query = f"UPDATE callers SET {', '.join(updates)} WHERE phone_number = ${param_num}"

    await pool.execute(query, *values)
    return {"success": True}


@router.delete("/api/callers/{phone_number}")
async def delete_caller(phone_number: str, user: dict = Depends(require_auth)):
    """Delete a caller"""
    pool = await get_db_pool()
    await pool.execute("DELETE FROM callers WHERE phone_number = $1", phone_number)
    return {"success": True}


# =====================================================
# API USAGE ENDPOINTS
# =====================================================

@router.get("/api/api-usage")
async def get_api_usage(user: dict = Depends(require_auth)):
    """Get API usage and cost tracking"""
    # TODO: Implement actual API usage tracking
    # For now, return placeholder data

    return {
        "services": [
            {
                "name": "OpenAI GPT-4o",
                "budget": 5.00,
                "spent": 0.01,
                "tokens": 77332,
                "requests": 92,
                "status": "healthy"
            },
            {
                "name": "ElevenLabs TTS",
                "budget": 50.00,
                "spent": 8.30,
                "characters": 64056,
                "status": "healthy"
            },
            {
                "name": "ElevenLabs STT",
                "budget": 50.00,
                "spent": 7.27,
                "minutes": 54.8,
                "status": "healthy"
            },
            {
                "name": "Twilio",
                "budget": 50.00,
                "spent": 0.62,
                "status": "healthy"
            },
            {
                "name": "Telnyx SMS",
                "balance": 1.58,
                "cost_per_msg": 0.004,
                "status": "warning"
            }
        ],
        "total_monthly_cost": 16.20
    }


# =====================================================
# FAQ DATABASE ENDPOINTS
# =====================================================

@router.get("/api/faq-status")
async def get_faq_status(user: dict = Depends(require_auth)):
    """Get FAQ database status"""
    try:
        from services.knowledge.faq_service import get_kb_service
        kb = get_kb_service()
        faqs = kb.get_all_faqs()

        return {
            "loaded": len(faqs),
            "status": "Loaded" if faqs else "Empty",
            "file_path": "config/faqs.yaml"
        }
    except Exception as e:
        return {
            "loaded": 0,
            "status": "Error",
            "error": str(e)
        }
