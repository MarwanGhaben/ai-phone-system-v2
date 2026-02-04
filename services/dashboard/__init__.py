"""
AI Voice Platform v2 - Dashboard Module
"""

from services.dashboard.dashboard_service import DashboardService, get_dashboard_service
from services.dashboard.auth_service import AuthService, get_auth_service
from services.dashboard.email_service import EmailService, get_email_service

__all__ = [
    "DashboardService",
    "get_dashboard_service",
    "AuthService",
    "get_auth_service",
    "EmailService",
    "get_email_service",
]
