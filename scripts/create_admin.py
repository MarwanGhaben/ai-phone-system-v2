#!/usr/bin/env python3
"""
=====================================================
AI Voice Platform v2 - Admin User Creation Script
=====================================================
Securely create an admin user with a strong password.

Usage:
    python scripts/create_admin.py

The script will prompt for username, email, and password.
Password must meet minimum security requirements.
"""

import asyncio
import getpass
import os
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import bcrypt
except ImportError:
    print("Error: bcrypt is required. Install with: pip install bcrypt")
    sys.exit(1)

try:
    import asyncpg
except ImportError:
    print("Error: asyncpg is required. Install with: pip install asyncpg")
    sys.exit(1)


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password meets security requirements.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"

    # Check for common weak passwords
    weak_passwords = [
        'password', 'admin', '123456', 'qwerty', 'letmein',
        'welcome', 'monkey', 'dragon', 'master', 'login'
    ]
    if password.lower() in weak_passwords or any(wp in password.lower() for wp in weak_passwords):
        return False, "Password contains common weak patterns"

    return True, ""


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


async def create_admin_user(
    database_url: str,
    username: str,
    email: str,
    password_hash: str
) -> bool:
    """
    Create admin user in database.

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = await asyncpg.connect(database_url)

        # Check if user already exists
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE username = $1",
            username
        )

        if existing:
            print(f"\nError: User '{username}' already exists.")
            await conn.close()
            return False

        # Insert new user
        await conn.execute("""
            INSERT INTO users (username, password_hash, email, role, is_active)
            VALUES ($1, $2, $3, 'admin', TRUE)
        """, username, password_hash, email)

        await conn.close()
        return True

    except asyncpg.PostgresError as e:
        print(f"\nDatabase error: {e}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("AI Voice Platform v2 - Admin User Creation")
    print("=" * 60)
    print()

    # Get database URL from environment or prompt
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("DATABASE_URL environment variable not set.")
        database_url = input("Enter PostgreSQL connection URL: ").strip()
        if not database_url:
            print("Error: Database URL is required.")
            sys.exit(1)

    print()
    print("Password requirements:")
    print("  - At least 12 characters")
    print("  - At least one uppercase letter")
    print("  - At least one lowercase letter")
    print("  - At least one digit")
    print("  - At least one special character")
    print()

    # Get username
    username = input("Username: ").strip()
    if not username:
        print("Error: Username is required.")
        sys.exit(1)

    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        print("Error: Username can only contain letters, numbers, and underscores.")
        sys.exit(1)

    # Get email
    email = input("Email: ").strip()
    if not email or '@' not in email:
        print("Error: Valid email is required.")
        sys.exit(1)

    # Get password (with confirmation)
    password = getpass.getpass("Password: ")
    is_valid, error_msg = validate_password(password)
    if not is_valid:
        print(f"Error: {error_msg}")
        sys.exit(1)

    password_confirm = getpass.getpass("Confirm password: ")
    if password != password_confirm:
        print("Error: Passwords do not match.")
        sys.exit(1)

    # Hash password
    print("\nHashing password...")
    password_hash = hash_password(password)

    # Create user
    print("Creating admin user...")
    success = asyncio.run(create_admin_user(
        database_url,
        username,
        email,
        password_hash
    ))

    if success:
        print()
        print("=" * 60)
        print(f"Admin user '{username}' created successfully!")
        print("=" * 60)
        print()
        print("You can now log in to the dashboard with these credentials.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
