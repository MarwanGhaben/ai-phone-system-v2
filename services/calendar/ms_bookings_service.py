"""
=====================================================
AI Voice Platform v2 - Microsoft Bookings Service
=====================================================
"""

import os
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from loguru import logger

from config.settings import get_settings
from services.calendar.calendar_base import (
    CalendarServiceBase,
    StaffMember,
    Service,
    TimeSlot,
    BookingResult
)


class MSBookingsService(CalendarServiceBase):
    """
    Microsoft Bookings integration via Microsoft Graph API

    Provides:
    - Staff member management
    - Service catalog
    - Availability checking
    - Appointment creation
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MS Bookings service

        Args:
            config: Configuration dict with tenant_id, client_id, client_secret, business_id
        """
        if config is None:
            settings = get_settings()
            config = {
                'tenant_id': settings.ms_bookings_tenant_id,
                'client_id': settings.ms_bookings_client_id,
                'client_secret': settings.ms_bookings_client_secret,
                'business_id': settings.ms_bookings_business_id
            }

        self.tenant_id = config.get('tenant_id', '')
        self.client_id = config.get('client_id', '')
        self.client_secret = config.get('client_secret', '')
        self.business_id = config.get('business_id', '')

        # Graph API URLs
        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.graph_url = "https://graph.microsoft.com/v1.0"

        # Caching
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._staff_cache: Dict[str, StaffMember] = {}
        self._services_cache: Dict[str, Service] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def is_available(self) -> bool:
        """Check if the service is configured and available"""
        return bool(self.tenant_id and self.client_id and
                   self.client_secret and self.business_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _get_access_token(self) -> str:
        """
        Get access token using client credentials flow

        Returns:
            Access token
        """
        # Check if cached token is still valid
        if self._access_token and self._token_expires_at:
            if datetime.now(timezone.utc) < self._token_expires_at:
                return self._access_token

        # Get new token
        client = await self._get_client()

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default"
        }

        try:
            response = await client.post(self.token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data['access_token']

            # Cache token expiration (subtract 5 minutes buffer)
            expires_in = token_data.get('expires_in', 3600)
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=expires_in - 300
            )

            logger.debug("MS Bookings: Obtained new access token")
            return self._access_token

        except Exception as e:
            logger.error(f"MS Bookings: Failed to get access token: {e}")
            raise

    async def _make_request(self, method: str, endpoint: str,
                           params: Dict = None, json_data: Dict = None) -> Optional[Dict]:
        """
        Make authenticated request to Graph API

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            Response JSON or None
        """
        try:
            token = await self._get_access_token()
            client = await self._get_client()

            url = f"{self.graph_url}{endpoint}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"MS Bookings: HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"MS Bookings: Request failed: {e}")
            return None

    async def get_staff_members(self) -> List[StaffMember]:
        """Get all staff members"""
        if self._staff_cache:
            return list(self._staff_cache.values())

        if not await self.is_available():
            logger.warning("MS Bookings: Not configured")
            return []

        try:
            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/staffMembers"
            result = await self._make_request("GET", endpoint)

            if result and 'value' in result:
                for item in result['value']:
                    staff = StaffMember(
                        id=item.get('id', ''),
                        name=item.get('displayName', ''),
                        email=item.get('emailAddress', ''),
                        role=item.get('role', '')
                    )
                    self._staff_cache[staff.name.lower()] = staff

                logger.info(f"MS Bookings: Loaded {len(self._staff_cache)} staff members")
                return list(self._staff_cache.values())

        except Exception as e:
            logger.error(f"MS Bookings: Failed to load staff: {e}")

        return []

    async def get_staff_by_name(self, name: str) -> Optional[StaffMember]:
        """Get staff member by name"""
        if not self._staff_cache:
            await self.get_staff_members()

        name_lower = name.lower().strip()

        # Direct match
        if name_lower in self._staff_cache:
            return self._staff_cache[name_lower]

        # Partial match
        for cached_name, staff in self._staff_cache.items():
            if name_lower in cached_name or cached_name in name_lower:
                return staff

        logger.warning(f"MS Bookings: Staff not found: {name}")
        return None

    async def get_services(self) -> List[Service]:
        """Get all services"""
        if self._services_cache:
            return list(self._services_cache.values())

        if not await self.is_available():
            logger.warning("MS Bookings: Not configured")
            return []

        try:
            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/services"
            result = await self._make_request("GET", endpoint)

            if result and 'value' in result:
                for item in result['value']:
                    # Parse duration from ISO 8601 format (e.g., PT30M)
                    duration_str = item.get('defaultDuration', 'PT30M')
                    duration_minutes = 30
                    if 'H' in duration_str:
                        hours = int(duration_str.replace('PT', '').replace('H', ''))
                        duration_minutes = hours * 60
                    elif 'M' in duration_str:
                        duration_minutes = int(duration_str.replace('PT', '').replace('M', ''))

                    service = Service(
                        id=item.get('id', ''),
                        name=item.get('displayName', ''),
                        description=item.get('description', ''),
                        duration_minutes=duration_minutes,
                        price=item.get('price', 0)
                    )
                    self._services_cache[service.id] = service

                logger.info(f"MS Bookings: Loaded {len(self._services_cache)} services")
                return list(self._services_cache.values())

        except Exception as e:
            logger.error(f"MS Bookings: Failed to load services: {e}")

        return []

    async def get_available_slots(self, service_id: str, staff_id: str = None,
                                 days_ahead: int = 7) -> List[TimeSlot]:
        """
        Get available time slots

        Args:
            service_id: Service ID
            staff_id: Optional staff ID to filter
            days_ahead: Number of days to look ahead

        Returns:
            List of available time slots
        """
        if not await self.is_available():
            logger.warning("MS Bookings: Not configured")
            return []

        try:
            # Ensure staff is loaded
            await self.get_staff_members()

            # Get staff IDs to query
            staff_ids = [staff_id] if staff_id else [s.id for s in self._staff_cache.values()]

            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/getStaffAvailability"

            # Use Eastern Time to match the business calendar timezone
            # This ensures slot times returned match local business hours
            windows_timezone = "Eastern Standard Time"

            # Use current time in a reasonable local approximation
            # We send local date range; MSGraph interprets in the specified timezone
            start_date = datetime.now() + timedelta(hours=1)
            end_date = start_date + timedelta(days=days_ahead)

            payload = {
                "startDateTime": {
                    "dateTime": start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    "timeZone": windows_timezone
                },
                "endDateTime": {
                    "dateTime": end_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    "timeZone": windows_timezone
                },
                "staffIds": staff_ids
            }

            result = await self._make_request("POST", endpoint, json_data=payload)

            slots = []
            slot_duration = 30  # minutes

            if result and 'value' in result:
                for staff_avail in result['value']:
                    sid = staff_avail.get('staffId', '')
                    staff_name = "Staff Member"
                    for s in self._staff_cache.values():
                        if s.id == sid:
                            staff_name = s.name
                            break

                    for item in staff_avail.get('availabilityItems', []):
                        if item.get('status') == 'available':
                            start_time_str = item.get('startDateTime', {}).get('dateTime', '')
                            end_time_str = item.get('endDateTime', {}).get('dateTime', '')

                            if start_time_str and end_time_str:
                                # Parse and strip timezone info to get naive datetime
                                # (times are already in Eastern since we requested Eastern timezone)
                                start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                                end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                                # Make naive for consistent comparison with user-provided times
                                start_dt = start_dt.replace(tzinfo=None)
                                end_dt = end_dt.replace(tzinfo=None)

                                # Generate 30-minute slots
                                current_slot = start_dt
                                while current_slot + timedelta(minutes=slot_duration) <= end_dt:
                                    slot_end = current_slot + timedelta(minutes=slot_duration)

                                    slots.append(TimeSlot(
                                        staff_id=sid,
                                        staff_name=staff_name,
                                        start_time=current_slot,
                                        end_time=slot_end,
                                        formatted=current_slot.strftime('%A, %B %d at %I:%M %p')
                                    ))

                                    current_slot = slot_end

            # Sort by start time
            slots.sort(key=lambda s: s.start_time)
            logger.info(f"MS Bookings: Found {len(slots)} available slots")
            return slots  # Return all slots — truncation here broke day matching

        except Exception as e:
            logger.error(f"MS Bookings: Failed to get availability: {e}")

        return []

    async def create_booking(self, service_id: str, staff_id: str,
                            start_time: datetime, customer_name: str,
                            customer_email: str, customer_phone: str = None,
                            notes: str = None) -> BookingResult:
        """
        Create a new booking

        Args:
            service_id: Service ID
            staff_id: Staff member ID
            start_time: Appointment start time
            customer_name: Customer name
            customer_email: Customer email
            customer_phone: Optional phone number
            notes: Optional notes

        Returns:
            BookingResult with success status
        """
        if not await self.is_available():
            return BookingResult(
                success=False,
                error_message="MS Bookings not configured"
            )

        try:
            # Get service for duration
            services = await self.get_services()
            service = next((s for s in services if s.id == service_id), None)
            duration = service.duration_minutes if service else 30

            end_time = start_time + timedelta(minutes=duration)

            # Windows timezone ID for Eastern Time
            windows_timezone = "Eastern Standard Time"

            payload = {
                "@odata.type": "#microsoft.graph.bookingAppointment",
                "serviceId": service_id,
                "staffMemberIds": [staff_id],
                "startDateTime": {
                    "@odata.type": "#microsoft.graph.dateTimeTimeZone",
                    "dateTime": start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    "timeZone": windows_timezone
                },
                "endDateTime": {
                    "@odata.type": "#microsoft.graph.dateTimeTimeZone",
                    "dateTime": end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    "timeZone": windows_timezone
                },
                "customerName": customer_name,
                "customerEmailAddress": customer_email,
                "customerPhone": customer_phone or "",
                "customerTimeZone": windows_timezone,
                "customerNotes": notes or "",
                "isLocationOnline": False,
                "optOutOfCustomerEmail": False
            }

            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/appointments"
            result = await self._make_request("POST", endpoint, json_data=payload)

            if result and result.get('id'):
                # Get staff name
                staff = await self.get_staff_by_name("")
                staff_name = "Staff Member"
                if staff_id:
                    for s in self._staff_cache.values():
                        if s.id == staff_id:
                            staff_name = s.name
                            break

                logger.info(f"MS Bookings: Created appointment for {customer_name}")

                return BookingResult(
                    success=True,
                    appointment_id=result.get('id'),
                    start_time=start_time.strftime('%A, %B %d at %I:%M %p'),
                    staff_name=staff_name,
                    customer_name=customer_name
                )
            else:
                return BookingResult(
                    success=False,
                    error_message="Failed to create appointment"
                )

        except Exception as e:
            logger.error(f"MS Bookings: Failed to create booking: {e}")
            return BookingResult(
                success=False,
                error_message=str(e)
            )

    async def close(self):
        """Close HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Global instance
_calendar_service: Optional[MSBookingsService] = None


def get_calendar_service() -> MSBookingsService:
    """Get global calendar service instance"""
    global _calendar_service
    if _calendar_service is None:
        settings = get_settings()
        config = {
            'tenant_id': settings.ms_bookings_tenant_id,
            'client_id': settings.ms_bookings_client_id,
            'client_secret': settings.ms_bookings_client_secret,
            'business_id': settings.ms_bookings_business_id
        }
        _calendar_service = MSBookingsService(config)
    return _calendar_service


def create_calendar_service(config: Dict[str, Any]) -> MSBookingsService:
    """
    Factory function to create calendar service

    Args:
        config: Configuration dict with tenant_id, client_id, client_secret, business_id

    Returns:
        MSBookingsService instance
    """
    return MSBookingsService(config)
