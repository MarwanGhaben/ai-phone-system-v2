"""
=====================================================
AI Voice Platform v2 - Email Service
=====================================================
Handles sending emails for MFA codes and notifications.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from loguru import logger
import os


class EmailService:
    """
    Email service for sending MFA codes and notifications

    Supports:
    - SMTP with TLS/SSL
    - Gmail, Outlook, custom SMTP servers
    """

    def __init__(self):
        # Load from environment variables
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.smtp_from = os.getenv('SMTP_FROM', self.smtp_user)
        self.smtp_use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'

        # Fallback: Use console logging if SMTP not configured
        self.is_configured = bool(self.smtp_user and self.smtp_password)

        if not self.is_configured:
            logger.warning("Email service not configured (SMTP_USER/SMTP_PASSWORD not set). MFA codes will be logged to console.")

    def _create_connection(self):
        """Create SMTP connection"""
        if self.smtp_use_tls:
            context = ssl.create_default_context()
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls(context=context)
        else:
            server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)

        server.login(self.smtp_user, self.smtp_password)
        return server

    def send_email(self, to_email: str, subject: str, body_html: str, body_text: str = None) -> bool:
        """
        Send an email

        Args:
            to_email: Recipient email address
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, for non-HTML clients)

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            logger.info(f"[EMAIL NOT CONFIGURED] Would send to {to_email}: {subject}")
            logger.info(f"[EMAIL BODY] {body_text or body_html}")
            return True  # Pretend success for testing

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_from
            msg['To'] = to_email

            # Add plain text and HTML versions
            if body_text:
                msg.attach(MIMEText(body_text, 'plain'))
            msg.attach(MIMEText(body_html, 'html'))

            server = self._create_connection()
            server.sendmail(self.smtp_from, to_email, msg.as_string())
            server.quit()

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_mfa_code(self, to_email: str, code: str, username: str = "User") -> bool:
        """
        Send MFA verification code email

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
        Send login notification email

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
