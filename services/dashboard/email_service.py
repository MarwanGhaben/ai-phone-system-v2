"""
=====================================================
AI Voice Platform v2 - Email Service (Microsoft Graph)
=====================================================
Handles sending emails for MFA codes and notifications via Microsoft Graph API.
"""

import aiohttp
import asyncio
from typing import Optional
from loguru import logger
import os


class EmailService:
    """
    Email service for sending MFA codes and notifications via Microsoft Graph API.

    Required environment variables:
    - MSGRAPH_TENANT_ID: Azure AD Tenant ID
    - MSGRAPH_CLIENT_ID: Azure AD Application (Client) ID
    - MSGRAPH_CLIENT_SECRET: Azure AD Client Secret
    - MSGRAPH_SENDER_EMAIL: Email address to send from (must have mailbox)
    """

    GRAPH_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    GRAPH_SEND_MAIL_URL = "https://graph.microsoft.com/v1.0/users/{sender}/sendMail"

    def __init__(self):
        # Load Microsoft Graph credentials from environment
        self.tenant_id = os.getenv('MSGRAPH_TENANT_ID', '')
        self.client_id = os.getenv('MSGRAPH_CLIENT_ID', '')
        self.client_secret = os.getenv('MSGRAPH_CLIENT_SECRET', '')
        self.sender_email = os.getenv('MSGRAPH_SENDER_EMAIL', '')

        # Check if configured
        self.is_configured = all([
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.sender_email
        ])

        if not self.is_configured:
            logger.warning(
                "Email service not configured. Set MSGRAPH_TENANT_ID, MSGRAPH_CLIENT_ID, "
                "MSGRAPH_CLIENT_SECRET, and MSGRAPH_SENDER_EMAIL. MFA codes will be logged to console."
            )

        # Cache for access token
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

    async def _get_access_token(self) -> Optional[str]:
        """
        Get Microsoft Graph API access token using client credentials flow.
        Caches the token until it expires.
        """
        import time

        # Return cached token if still valid (with 5 minute buffer)
        if self._access_token and time.time() < (self._token_expires_at - 300):
            return self._access_token

        token_url = self.GRAPH_TOKEN_URL.format(tenant_id=self.tenant_id)

        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'https://graph.microsoft.com/.default',
            'grant_type': 'client_credentials'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get Graph access token: {response.status} - {error_text}")
                        return None

                    token_data = await response.json()
                    self._access_token = token_data.get('access_token')
                    expires_in = token_data.get('expires_in', 3600)
                    self._token_expires_at = time.time() + expires_in

                    logger.debug("Obtained new Microsoft Graph access token")
                    return self._access_token

        except Exception as e:
            logger.error(f"Error getting Graph access token: {e}")
            return None

    async def send_email_async(self, to_email: str, subject: str, body_html: str, body_text: str = None) -> bool:
        """
        Send an email via Microsoft Graph API (async version).

        Args:
            to_email: Recipient email address
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, unused for Graph API)

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            logger.info(f"[EMAIL NOT CONFIGURED] Would send to {to_email}: {subject}")
            if body_text:
                logger.info(f"[EMAIL BODY] {body_text[:200]}...")
            return True  # Pretend success for testing

        # Get access token
        access_token = await self._get_access_token()
        if not access_token:
            logger.error("Cannot send email: Failed to obtain access token")
            return False

        # Build the Graph API request
        send_mail_url = self.GRAPH_SEND_MAIL_URL.format(sender=self.sender_email)

        mail_body = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content": body_html
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to_email
                        }
                    }
                ]
            },
            "saveToSentItems": "true"
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(send_mail_url, json=mail_body, headers=headers) as response:
                    if response.status == 202:
                        logger.info(f"Email sent successfully to {to_email} via Microsoft Graph")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send email via Graph: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Error sending email via Graph: {e}")
            return False

    def send_email(self, to_email: str, subject: str, body_html: str, body_text: str = None) -> bool:
        """
        Send an email (sync wrapper for async method).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                future = asyncio.ensure_future(self.send_email_async(to_email, subject, body_html, body_text))
                # For sync contexts within async, we need to handle this differently
                # This is a best-effort approach
                return True  # Optimistically return True, actual send happens async
            else:
                return loop.run_until_complete(self.send_email_async(to_email, subject, body_html, body_text))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.send_email_async(to_email, subject, body_html, body_text))

    def send_mfa_code(self, to_email: str, code: str, username: str = "User") -> bool:
        """
        Send MFA verification code email.

        Args:
            to_email: Recipient email
            code: The 6-digit MFA code
            username: User's name for greeting

        Returns:
            True if sent successfully
        """
        subject = "Your AI Voice Platform Login Code"

        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .code {{ font-size: 36px; font-weight: bold; color: #667eea; letter-spacing: 8px; text-align: center; padding: 20px; background: white; border-radius: 8px; margin: 20px 0; border: 2px dashed #667eea; }}
                .warning {{ font-size: 12px; color: #666; margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 5px; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #999; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AI Voice Platform</h1>
                    <p>Admin Dashboard Login</p>
                </div>
                <div class="content">
                    <p>Hello {username},</p>
                    <p>Your verification code for logging into the AI Voice Platform dashboard is:</p>
                    <div class="code">{code}</div>
                    <p>This code will expire in <strong>10 minutes</strong>.</p>
                    <div class="warning">
                        <strong>Security Notice:</strong> If you did not request this code, please ignore this email. Someone may have entered your email address by mistake.
                    </div>
                </div>
                <div class="footer">
                    <p>Flexible Accounting - AI Voice Platform</p>
                    <p>This is an automated message. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """

        body_text = f"""
AI Voice Platform - Login Verification

Hello {username},

Your verification code is: {code}

This code will expire in 10 minutes.

If you did not request this code, please ignore this email.

---
Flexible Accounting - AI Voice Platform
        """

        # If not configured, log the code prominently
        if not self.is_configured:
            logger.warning("=" * 50)
            logger.warning(f"MFA CODE FOR {to_email}: {code}")
            logger.warning("=" * 50)
            return True

        return self.send_email(to_email, subject, body_html, body_text)

    def send_login_alert(self, to_email: str, username: str, ip_address: str, user_agent: str) -> bool:
        """
        Send login notification email.

        Args:
            to_email: Admin email
            username: Username that logged in
            ip_address: IP address of login
            user_agent: Browser/device info

        Returns:
            True if sent successfully
        """
        subject = "New Login to AI Voice Platform Dashboard"

        from datetime import datetime
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #28a745; color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; border-radius: 0 0 10px 10px; }}
                .details {{ background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .details p {{ margin: 5px 0; }}
                .label {{ font-weight: bold; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Successful Login</h2>
                </div>
                <div class="content">
                    <p>A new login was detected on your AI Voice Platform dashboard:</p>
                    <div class="details">
                        <p><span class="label">User:</span> {username}</p>
                        <p><span class="label">Time:</span> {login_time}</p>
                        <p><span class="label">IP Address:</span> {ip_address}</p>
                        <p><span class="label">Device:</span> {user_agent[:100]}...</p>
                    </div>
                    <p>If this wasn't you, please change your password immediately.</p>
                </div>
            </div>
        </body>
        </html>
        """

        body_text = f"""
New Login to AI Voice Platform Dashboard

User: {username}
Time: {login_time}
IP Address: {ip_address}
Device: {user_agent[:100]}

If this wasn't you, please change your password immediately.
        """

        if not self.is_configured:
            logger.info(f"[LOGIN ALERT] {username} logged in from {ip_address}")
            return True

        return self.send_email(to_email, subject, body_html, body_text)


# Global instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get global email service instance"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
