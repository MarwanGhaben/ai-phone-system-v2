"""
=====================================================
AI Voice Platform v2 - Dashboard Service
=====================================================
"""

from datetime import datetime
from typing import Dict, Optional
from loguru import logger
import os


class DashboardService:
    """
    Dashboard service for admin interface

    Provides metrics and statistics for the admin dashboard.
    """

    def __init__(self, db=None):
        """
        Initialize dashboard service

        Args:
            db: Database service (optional)
        """
        self.db = db

    # ============================================================
    # OPERATIONAL OVERVIEW
    # ============================================================

    async def get_operational_overview(self) -> Dict:
        """
        Get at-a-glance operational status

        Returns:
            {
                'active_calls': {...},
                'system_health': {...},
                'today_bookings': {...},
                'error_rate': {...}
            }
        """
        try:
            # Active calls
            active_calls = self._get_active_calls_info()

            # System health
            system_health = await self._get_system_health()

            # Today's bookings
            today_bookings = await self._get_today_bookings()

            # Error rate
            error_rate = self._get_error_rate()

            return {
                'active_calls': active_calls,
                'system_health': system_health,
                'today_bookings': today_bookings,
                'error_rate': error_rate
            }

        except Exception as e:
            logger.error(f"Dashboard: Error getting operational overview: {e}")
            return self._get_empty_operational_overview()

    def _get_active_calls_info(self) -> Dict:
        """Get active calls information"""
        from services.conversation.orchestrator import get_orchestrator

        try:
            orchestrator = get_orchestrator()
            active_count = len(orchestrator._conversations)
        except:
            active_count = 0

        # Get worker count
        worker_count = int(os.getenv('WEB_CONCURRENCY', '4'))

        if worker_count == 1:
            worker_status = 'safe mode'
        elif worker_count <= 4:
            worker_status = 'optimal'
        else:
            worker_status = 'high capacity'

        return {
            'total': active_count,
            'worker_count': worker_count,
            'worker_status': worker_status
        }

    async def _get_system_health(self) -> Dict:
        """Get overall system health percentage"""
        try:
            import psutil

            # CPU health (interval=0.1 blocks briefly but gives accurate reading)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = 'healthy' if cpu_percent < 70 else 'warning' if cpu_percent < 90 else 'danger'

            # Memory health
            memory = psutil.virtual_memory()
            memory_status = 'healthy' if memory.percent < 70 else 'warning' if memory.percent < 90 else 'danger'
            memory_str = f"{memory.used / (1024**3):.1f} / {memory.total / (1024**3):.0f} GB"

            # Disk health
            disk = psutil.disk_usage('/')
            disk_status = 'healthy' if disk.percent < 70 else 'warning' if disk.percent < 90 else 'danger'
            disk_str = f"{disk.percent:.0f}% / {disk.total / (1024**3):.0f} GB"

            # Calculate overall score
            cpu_score = max(0, 100 - cpu_percent)
            memory_score = max(0, 100 - memory.percent)
            disk_score = max(0, 100 - disk.percent)
            overall_score = (cpu_score + memory_score + disk_score) / 3

            if overall_score >= 90:
                status = 'operational'
            elif overall_score >= 70:
                status = 'degraded'
            else:
                status = 'critical'

            return {
                'overall_percentage': round(overall_score, 0),
                'status': status,
                'components': [
                    {'name': 'CPU', 'score': round(cpu_score, 0), 'value': f"{cpu_percent:.0f}%", 'status': cpu_status},
                    {'name': 'Memory', 'score': round(memory_score, 0), 'value': memory_str, 'status': memory_status},
                    {'name': 'Disk', 'score': round(disk_score, 0), 'value': disk_str, 'status': disk_status}
                ]
            }
        except Exception as e:
            logger.error(f"Dashboard: Error getting system health: {e}")
            return {'overall_percentage': 0, 'status': 'unknown', 'components': []}

    async def _get_today_bookings(self) -> Dict:
        """Report unavailable booking metrics for the legacy overview API."""
        return {
            'available': False,
            'total': None,
            'confirmed': None,
            'pending': None,
            'by_accountant': None,
        }

    def _get_error_rate(self) -> Dict:
        """Report that error-rate telemetry is not instrumented."""
        return {
            'available': False,
            'current': None,
            'threshold': None,
            'status': 'not_tracked',
        }

    def _get_empty_operational_overview(self) -> Dict:
        """Return empty operational overview when data unavailable"""
        return {
            'active_calls': {'total': 0, 'worker_count': 1, 'worker_status': 'unknown'},
            'system_health': {'overall_percentage': 0, 'status': 'unknown', 'components': []},
            'today_bookings': {
                'available': False,
                'total': None,
                'confirmed': None,
                'pending': None,
                'by_accountant': None,
            },
            'error_rate': self._get_error_rate(),
        }

    # ============================================================
    # CALL ANALYTICS
    # ============================================================

    async def get_call_analytics(self) -> Dict:
        """
        Get caller analytics and insights

        Returns:
            {
                'total_calls': 1247,
                'language_distribution': [...],
                'quality_metrics': {...},
                'recent_calls': [...]
            }
        """
        from services.dashboard.metrics_service import fetch_call_statistics
        from services.database import get_db_pool

        statistics = await fetch_call_statistics(await get_db_pool())
        statistics.update({
            'available': True,
            'quality_metrics': {
                'interruptions': None,
                'transfer_rate': statistics['transfer_rate'],
                'abandoned_calls': None,
            },
            'recent_calls': []
        })
        return statistics

    # ============================================================
    # API USAGE MONITORING
    # ============================================================

    async def get_api_monitoring(self) -> Dict:
        """
        Get API usage monitoring

        Returns:
            {
                'services': [...]
            }
        """
        return {
            'available': False,
            'message': 'Cost tracking is not instrumented',
            'services': [
                {
                    'name': 'Twilio',
                    'icon': '',
                    'spent': None,
                    'budget': None,
                    'percentage': None,
                    'status': 'not_tracked'
                },
                {
                    'name': 'Deepgram',
                    'icon': '',
                    'spent': None,
                    'budget': None,
                    'percentage': None,
                    'status': 'not_tracked'
                },
                {
                    'name': 'ElevenLabs',
                    'icon': '',
                    'spent': None,
                    'budget': None,
                    'percentage': None,
                    'status': 'not_tracked'
                },
                {
                    'name': 'OpenAI',
                    'icon': '',
                    'spent': None,
                    'budget': None,
                    'percentage': None,
                    'status': 'not_tracked'
                }
            ]
        }

    # ============================================================
    # SYSTEM HEALTH
    # ============================================================

    async def get_system_health(self) -> Dict:
        """
        Get consolidated system health metrics

        Returns:
            {
                'uptime': '15d 4h 23m',
                'metrics': [...]
            }
        """
        try:
            import psutil

            # Uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"

            health = await self._get_system_health()

            # Worker count
            worker_count = int(os.getenv('WEB_CONCURRENCY', '4'))
            if worker_count == 1:
                worker_str = '1 (safe mode)'
                worker_status = 'safe'
            elif worker_count <= 4:
                worker_str = f'{worker_count} (optimal)'
                worker_status = 'optimal'
            else:
                worker_str = f'{worker_count} (high capacity)'
                worker_status = 'high'

            return {
                'uptime': uptime_str,
                'metrics': health['components'] + [
                    {
                        'name': 'Workers',
                        'current': worker_str,
                        'status': worker_status,
                        'trend': 'stable'
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Dashboard: Error getting system health: {e}")
            return {'uptime': 'Unknown', 'metrics': []}

    # ============================================================
    # BOOKINGS OVERVIEW
    # ============================================================

    async def get_bookings_overview(self) -> Dict:
        """
        Get booking management data

        Returns:
            {
                'stats': {...},
                'recent_bookings': [...],
                'upcoming_bookings': [...]
            }
        """
        return {
            'available': False,
            'message': 'Use /dashboard/api/bookings for persisted booking data',
            'stats': {
                'total': None,
                'upcoming': None,
                'today': None,
                'by_accountant': None,
            },
            'recent_bookings': [],
            'upcoming_bookings': []
        }


# Global instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get global dashboard service instance"""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
