"""
=====================================================
AI Voice Platform v2 - Security Middleware
=====================================================
Security utilities including Twilio signature validation,
rate limiting, and security headers.
"""

import time
import hashlib
from collections import defaultdict
from typing import Callable, Dict, Optional
from urllib.parse import urlencode

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.request_validator import RequestValidator
from loguru import logger


class TwilioSignatureValidator:
    """
    Validates Twilio webhook request signatures.

    Twilio signs all webhook requests with X-Twilio-Signature header.
    This prevents attackers from sending fake webhook requests.
    """

    def __init__(self, auth_token: str):
        """
        Initialize validator with Twilio auth token.

        Args:
            auth_token: Twilio account auth token
        """
        self.validator = RequestValidator(auth_token)

    async def validate_request(
        self,
        request: Request,
        url: Optional[str] = None
    ) -> bool:
        """
        Validate a Twilio webhook request.

        Args:
            request: FastAPI request object
            url: Optional URL override (use if behind reverse proxy)

        Returns:
            True if signature is valid, False otherwise
        """
        # Get the signature from header
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature:
            logger.warning("Twilio: Missing X-Twilio-Signature header")
            return False

        # Build the URL that Twilio signed
        if url:
            request_url = url
        else:
            # Use X-Forwarded headers if behind reverse proxy
            proto = request.headers.get("X-Forwarded-Proto", request.url.scheme)
            host = request.headers.get("X-Forwarded-Host", request.headers.get("Host", ""))
            request_url = f"{proto}://{host}{request.url.path}"

        # Get POST params (for POST requests) or query params (for GET)
        if request.method == "POST":
            # For form data
            try:
                form_data = await request.form()
                params = dict(form_data)
            except Exception:
                params = {}
        else:
            # For GET requests, use query params
            params = dict(request.query_params)

        # Validate the signature
        is_valid = self.validator.validate(request_url, params, signature)

        if not is_valid:
            logger.warning(
                f"Twilio: Invalid signature for {request_url}. "
                f"Expected valid signature for params: {list(params.keys())}"
            )

        return is_valid


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    For production, consider using Redis for distributed rate limiting.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests allowed per minute per IP
            burst_limit: Max requests allowed in quick succession
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, list] = defaultdict(list)
        self.burst_tracker: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling reverse proxies."""
        # Check for forwarded IP (reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # Take first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP (nginx)
        real_ip = request.headers.get("X-Real-IP", "")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_ip: str, window_seconds: int = 60):
        """Remove requests outside the time window."""
        now = time.time()
        cutoff = now - window_seconds

        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > cutoff
        ]

        # Cleanup burst tracker (1 second window)
        burst_cutoff = now - 1
        self.burst_tracker[client_ip] = [
            t for t in self.burst_tracker[client_ip] if t > burst_cutoff
        ]

    def is_rate_limited(self, request: Request) -> tuple[bool, dict]:
        """
        Check if request should be rate limited.

        Returns:
            Tuple of (is_limited, info_dict)
        """
        client_ip = self._get_client_ip(request)
        now = time.time()

        self._cleanup_old_requests(client_ip)

        # Check burst limit (requests per second)
        if len(self.burst_tracker[client_ip]) >= self.burst_limit:
            return True, {
                "reason": "burst_limit_exceeded",
                "limit": self.burst_limit,
                "retry_after": 1
            }

        # Check requests per minute
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            oldest = min(self.requests[client_ip]) if self.requests[client_ip] else now
            retry_after = int(60 - (now - oldest)) + 1
            return True, {
                "reason": "rate_limit_exceeded",
                "limit": self.requests_per_minute,
                "retry_after": retry_after
            }

        # Record this request
        self.requests[client_ip].append(now)
        self.burst_tracker[client_ip].append(now)

        return False, {}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # HSTS - force HTTPS (only if request came over HTTPS)
        if request.headers.get("X-Forwarded-Proto") == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection (legacy, but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Skip ALL restrictive headers for dashboard (it needs inline scripts, fonts, etc.)
        if request.url.path.startswith("/dashboard"):
            # No CSP or X-Frame-Options for dashboard - let it load freely
            pass
        else:
            # Strict headers for API endpoints only
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self' wss: ws:;"
            )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        exclude_paths: list[str] = None
    ):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute, burst_limit)
        self.exclude_paths = exclude_paths or ["/health", "/ws/calls"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        is_limited, info = self.limiter.is_rate_limited(request)

        if is_limited:
            logger.warning(
                f"Rate limit exceeded for {request.client.host}: {info['reason']}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too many requests",
                    "retry_after": info.get("retry_after", 60)
                },
                headers={"Retry-After": str(info.get("retry_after", 60))}
            )

        return await call_next(request)


# Dependency for Twilio webhook endpoints
async def validate_twilio_signature(request: Request) -> bool:
    """
    FastAPI dependency to validate Twilio webhook signatures.

    Usage:
        @app.post("/api/incoming-call")
        async def incoming_call(request: Request, _: bool = Depends(validate_twilio_signature)):
            ...
    """
    from config.settings import get_settings
    settings = get_settings()

    # Skip validation in development mode if explicitly disabled
    if settings.environment == "development":
        skip_validation = getattr(settings, 'skip_twilio_signature_validation', False)
        if skip_validation:
            logger.warning("Twilio: Signature validation SKIPPED (development mode)")
            return True

    validator = TwilioSignatureValidator(settings.twilio_auth_token)

    # Build the public URL for signature validation
    # Twilio signs against the public URL, not internal container URL
    if settings.public_domain:
        proto = "https"
        host = settings.public_domain
        public_url = f"{proto}://{host}{request.url.path}"
    else:
        public_url = None  # Will use request URL

    is_valid = await validator.validate_request(request, public_url)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio signature"
        )

    return True
