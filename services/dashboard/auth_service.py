"""
=====================================================
AI Voice Platform v2 - Authentication Service
=====================================================
Handles login, MFA, and session management for the admin dashboard.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from loguru import logger

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available, using fallback hashing")

from services.database import get_db_pool


class AuthService:
    """
    Authentication service for admin dashboard

    Features:
    - Username/password login with bcrypt
    - MFA via email (6-digit code)
    - Session token management
    - Rate limiting for login attempts
    """

    MFA_CODE_LENGTH = 6
    MFA_CODE_EXPIRY_MINUTES = 10
    SESSION_EXPIRY_HOURS = 24
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_MINUTES = 15

    def __init__(self):
        self._login_attempts: Dict[str, list] = {}  # IP -> [timestamps]

    # ============================================================
    # PASSWORD HANDLING
    # ============================================================

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Fallback: SHA256 with salt (less secure but works without bcrypt)
            salt = secrets.token_hex(16)
            hash_val = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return f"sha256:{salt}:{hash_val}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        if password_hash.startswith("sha256:"):
            # Fallback hash
            parts = password_hash.split(":")
            if len(parts) != 3:
                return False
            salt, stored_hash = parts[1], parts[2]
            computed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return secrets.compare_digest(computed, stored_hash)
        elif BCRYPT_AVAILABLE:
            try:
                return bcrypt.checkpw(password.encode(), password_hash.encode())
            except Exception:
                return False
        return False

    # ============================================================
    # RATE LIMITING
    # ============================================================

    def _check_rate_limit(self, ip_address: str) -> Tuple[bool, int]:
        """
        Check if IP is rate limited

        Returns:
            (is_allowed, seconds_remaining)
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.LOCKOUT_MINUTES)

        # Clean old attempts
        if ip_address in self._login_attempts:
            self._login_attempts[ip_address] = [
                ts for ts in self._login_attempts[ip_address]
                if ts > cutoff
            ]

        attempts = self._login_attempts.get(ip_address, [])

        if len(attempts) >= self.MAX_LOGIN_ATTEMPTS:
            oldest = min(attempts)
            unlock_time = oldest + timedelta(minutes=self.LOCKOUT_MINUTES)
            remaining = (unlock_time - now).total_seconds()
            return False, max(0, int(remaining))

        return True, 0

    def _record_login_attempt(self, ip_address: str):
        """Record a failed login attempt"""
        if ip_address not in self._login_attempts:
            self._login_attempts[ip_address] = []
        self._login_attempts[ip_address].append(datetime.now())

    def _clear_login_attempts(self, ip_address: str):
        """Clear login attempts after successful login"""
        if ip_address in self._login_attempts:
            del self._login_attempts[ip_address]

    # ============================================================
    # LOGIN & USER LOOKUP
    # ============================================================

    async def authenticate_user(self, username: str, password: str, ip_address: str = "") -> Tuple[bool, Optional[Dict], str]:
        """
        Authenticate user with username and password

        Returns:
            (success, user_dict, error_message)
        """
        # Check rate limit
        allowed, wait_seconds = self._check_rate_limit(ip_address)
        if not allowed:
            return False, None, f"Too many login attempts. Try again in {wait_seconds} seconds."

        pool = await get_db_pool()

        # Look up user
        row = await pool.fetchrow(
            """
            SELECT id, username, email, password_hash, is_active, is_superuser
            FROM admin_users
            WHERE username = $1 OR email = $1
            """,
            username
        )

        if not row:
            self._record_login_attempt(ip_address)
            return False, None, "Invalid username or password"

        if not row['is_active']:
            return False, None, "Account is disabled"

        # Verify password
        if not self.verify_password(password, row['password_hash']):
            self._record_login_attempt(ip_address)
            return False, None, "Invalid username or password"

        # Success - clear attempts
        self._clear_login_attempts(ip_address)

        user = {
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'is_superuser': row['is_superuser']
        }

        return True, user, ""

    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        pool = await get_db_pool()
        row = await pool.fetchrow(
            "SELECT id, username, email, is_active, is_superuser FROM admin_users WHERE id = $1",
            user_id
        )
        if row:
            return {
                'id': row['id'],
                'username': row['username'],
                'email': row['email'],
                'is_active': row['is_active'],
                'is_superuser': row['is_superuser']
            }
        return None

    # ============================================================
    # MFA CODE HANDLING
    # ============================================================

    def generate_mfa_code(self) -> str:
        """Generate a 6-digit MFA code"""
        return ''.join(secrets.choice('0123456789') for _ in range(self.MFA_CODE_LENGTH))

    async def create_mfa_code(self, user_id: int) -> str:
        """
        Create and store a new MFA code for user

        Returns:
            The generated code
        """
        code = self.generate_mfa_code()
        expires_at = datetime.now() + timedelta(minutes=self.MFA_CODE_EXPIRY_MINUTES)

        pool = await get_db_pool()

        # Invalidate any existing codes for this user
        await pool.execute(
            "UPDATE mfa_codes SET used = TRUE WHERE user_id = $1 AND used = FALSE",
            user_id
        )

        # Create new code
        await pool.execute(
            """
            INSERT INTO mfa_codes (user_id, code, expires_at)
            VALUES ($1, $2, $3)
            """,
            user_id, code, expires_at
        )

        logger.info(f"MFA code created for user {user_id}")
        return code

    async def verify_mfa_code(self, user_id: int, code: str) -> Tuple[bool, str]:
        """
        Verify an MFA code

        Returns:
            (success, error_message)
        """
        pool = await get_db_pool()

        row = await pool.fetchrow(
            """
            SELECT id, expires_at
            FROM mfa_codes
            WHERE user_id = $1 AND code = $2 AND used = FALSE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            user_id, code
        )

        if not row:
            return False, "Invalid verification code"

        if row['expires_at'] < datetime.now():
            return False, "Verification code has expired"

        # Mark code as used
        await pool.execute(
            "UPDATE mfa_codes SET used = TRUE WHERE id = $1",
            row['id']
        )

        logger.info(f"MFA code verified for user {user_id}")
        return True, ""

    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================

    def generate_session_token(self) -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(32)

    async def create_session(self, user_id: int, ip_address: str = "", user_agent: str = "") -> str:
        """
        Create a new session for user

        Returns:
            Session token
        """
        token = self.generate_session_token()
        expires_at = datetime.now() + timedelta(hours=self.SESSION_EXPIRY_HOURS)

        pool = await get_db_pool()

        await pool.execute(
            """
            INSERT INTO admin_sessions (user_id, session_token, expires_at, ip_address, user_agent)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user_id, token, expires_at, ip_address, user_agent
        )

        # Update last login
        await pool.execute(
            "UPDATE admin_users SET last_login = NOW() WHERE id = $1",
            user_id
        )

        logger.info(f"Session created for user {user_id}")
        return token

    async def validate_session(self, token: str) -> Optional[Dict]:
        """
        Validate a session token

        Returns:
            User dict if valid, None otherwise
        """
        pool = await get_db_pool()

        row = await pool.fetchrow(
            """
            SELECT s.user_id, s.expires_at, u.username, u.email, u.is_active, u.is_superuser
            FROM admin_sessions s
            JOIN admin_users u ON s.user_id = u.id
            WHERE s.session_token = $1
            """,
            token
        )

        if not row:
            return None

        if row['expires_at'] < datetime.now():
            # Session expired - clean it up
            await pool.execute(
                "DELETE FROM admin_sessions WHERE session_token = $1",
                token
            )
            return None

        if not row['is_active']:
            return None

        return {
            'id': row['user_id'],
            'username': row['username'],
            'email': row['email'],
            'is_superuser': row['is_superuser']
        }

    async def invalidate_session(self, token: str):
        """Invalidate (logout) a session"""
        pool = await get_db_pool()
        await pool.execute(
            "DELETE FROM admin_sessions WHERE session_token = $1",
            token
        )
        logger.info("Session invalidated")

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and MFA codes"""
        pool = await get_db_pool()

        # Clean expired sessions
        await pool.execute(
            "DELETE FROM admin_sessions WHERE expires_at < NOW()"
        )

        # Clean expired MFA codes
        await pool.execute(
            "DELETE FROM mfa_codes WHERE expires_at < NOW() OR used = TRUE"
        )

    # ============================================================
    # USER MANAGEMENT
    # ============================================================

    async def create_user(self, username: str, email: str, password: str, is_superuser: bool = False) -> Tuple[bool, str]:
        """
        Create a new admin user

        Returns:
            (success, error_message)
        """
        pool = await get_db_pool()

        # Check if username or email already exists
        existing = await pool.fetchval(
            "SELECT 1 FROM admin_users WHERE username = $1 OR email = $2",
            username, email
        )

        if existing:
            return False, "Username or email already exists"

        password_hash = self.hash_password(password)

        await pool.execute(
            """
            INSERT INTO admin_users (username, email, password_hash, is_superuser)
            VALUES ($1, $2, $3, $4)
            """,
            username, email, password_hash, is_superuser
        )

        logger.info(f"Admin user created: {username}")
        return True, ""

    async def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        pool = await get_db_pool()
        password_hash = self.hash_password(new_password)

        await pool.execute(
            "UPDATE admin_users SET password_hash = $1, updated_at = NOW() WHERE id = $2",
            password_hash, user_id
        )

        # Invalidate all existing sessions
        await pool.execute(
            "DELETE FROM admin_sessions WHERE user_id = $1",
            user_id
        )

        logger.info(f"Password updated for user {user_id}")
        return True


# Global instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get global auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
