"""
=====================================================
AI Voice Platform v2 - Security Middleware
=====================================================
Security utilities including Twilio signature validation,
rate limiting, and security headers.
"""

from typing import Callable, Optional

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from twilio.request_validator import RequestValidator
from loguru import logger
from redis.exceptions import RedisError

from services.security.client_ip import get_client_ip
from services.security.rate_limiter import RedisRateLimiter, create_redis_client


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
        self.limiter = RedisRateLimiter(
            create_redis_client(), requests_per_minute, burst_limit
        )
        self.exclude_paths = exclude_paths or ["/health", "/ws/calls"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        client_ip = get_client_ip(request)
        try:
            is_limited, info = await self.limiter.check(client_ip)
        except RedisError as exc:
            logger.error(f"Rate limiting unavailable: {exc}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Rate limiting temporarily unavailable"},
            )

        if is_limited:
            logger.warning(
                f"Rate limit exceeded for {client_ip}: {info['reason']}"
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
