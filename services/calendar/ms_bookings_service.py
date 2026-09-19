"""
=====================================================
AI Voice Platform v2 - Microsoft Bookings Service
=====================================================
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import math
from typing import Optional, List, Dict, Any
from urllib.parse import quote

import httpx
from loguru import logger

from config.settings import get_settings
from services.calendar.business_time import business_now, parse_graph_datetime
from services.calendar.contracts import decode_availability
from services.calendar.service_facts import ServiceFactsRead, decode_service_facts
from services.calendar.calendar_base import (
    CalendarServiceBase,
    StaffMember,
    Service,
    TimeSlot,
    BookingResult
)
from services.scheduling.models import (
    AvailabilityContractError,
    AvailabilityFailure,
    AvailabilityFailureCategory,
    AvailabilityQuery,
    AvailabilityResult,
    AvailabilityStatus,
    CalendarScope,
)


_TOKEN_RESPONSE_LIMIT = 64 * 1024
_AVAILABILITY_RESPONSE_LIMIT = 1024 * 1024
_CONTRACT_ERROR = "invalid availability contract"


class _AvailabilityReadFailure(Exception):
    """A safe fixed failure crossing the typed availability transport."""

    def __init__(self, category: AvailabilityFailureCategory,
                 http_status: int | None = None):
        super().__init__(category.value)
        self.category = category
        self.http_status = http_status


class _AvailabilityPermit:
    """Tracks whether a retained cleanup owns an availability admission slot."""

    def __init__(self) -> None:
        self.acquired = False
        self.handed_off = False


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _availability_http_failure(status: int) -> _AvailabilityReadFailure:
    if status == 401:
        category = AvailabilityFailureCategory.AUTHENTICATION
    elif status == 403:
        category = AvailabilityFailureCategory.PERMISSION_DENIED
    elif status == 429:
        category = AvailabilityFailureCategory.THROTTLED
    elif 200 <= status < 300:
        category = AvailabilityFailureCategory.INVALID_RESPONSE
    else:
        category = AvailabilityFailureCategory.PROVIDER_ERROR
    return _AvailabilityReadFailure(category, status)


def _availability_token_http_failure(status: int) -> _AvailabilityReadFailure:
    if status == 400:
        return _AvailabilityReadFailure(
            AvailabilityFailureCategory.AUTHENTICATION,
            status,
        )
    return _availability_http_failure(status)


def normalize_customer_phone(phone_number: str) -> Optional[str]:
    digits = "".join(character for character in phone_number if character.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def appointment_customer_phone(appointment: Dict[str, Any]) -> str:
    direct_phone = appointment.get("customerPhone", "")
    if direct_phone:
        return direct_phone
    customers = appointment.get("customers", [])
    return customers[0].get("phone", "") if customers else ""


class MSBookingsService(CalendarServiceBase):
    """
    Microsoft Bookings integration via Microsoft Graph API

    Provides:
    - Staff member management
    - Service catalog
    - Availability checking
    - Appointment creation
    """

    AVAILABILITY_TIMEOUT_SECONDS = 20.0
    AVAILABILITY_CLEANUP_GRACE_SECONDS = 1.0
    AVAILABILITY_CLEANUP_CAPACITY = 8

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
        self._availability_slots = asyncio.Semaphore(
            self.AVAILABILITY_CLEANUP_CAPACITY
        )
        self._availability_cleanup_tasks: set[asyncio.Task[bytes]] = set()

    async def is_available(self) -> bool:
        """Check if the service is configured and available"""
        return bool(self.tenant_id and self.client_id and
                   self.client_secret and self.business_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=False,
            )
        return self._http_client

    @staticmethod
    async def _read_availability_body(
        response: httpx.Response,
        limit: int,
    ) -> bytes:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > limit:
                raise _AvailabilityReadFailure(
                    AvailabilityFailureCategory.INCOMPLETE,
                    response.status_code,
                )
            body.extend(chunk)
        return bytes(body)

    async def _read_streamed_availability_response(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        limit: int,
        *,
        token_boundary: bool = False,
        **request_kwargs: Any,
    ) -> bytes:
        async with client.stream(
            method,
            url,
            follow_redirects=False,
            **request_kwargs,
        ) as response:
            if response.status_code != 200:
                classifier = (
                    _availability_token_http_failure
                    if token_boundary
                    else _availability_http_failure
                )
                raise classifier(response.status_code)
            return await self._read_availability_body(response, limit)

    def _availability_cleanup_finished(
        self,
        task: asyncio.Task[bytes],
    ) -> None:
        self._availability_cleanup_tasks.discard(task)
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        self._availability_slots.release()

    def _retain_availability_cleanup(
        self,
        task: asyncio.Task[bytes],
        permit: _AvailabilityPermit,
    ) -> None:
        permit.handed_off = True
        self._availability_cleanup_tasks.add(task)
        task.add_done_callback(self._availability_cleanup_finished)

    async def _run_availability_io(
        self,
        operation: Any,
        permit: _AvailabilityPermit,
    ) -> bytes:
        task = asyncio.create_task(
            operation,
            name="ms_bookings_availability_io",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            self._retain_availability_cleanup(task, permit)
            try:
                await asyncio.wait(
                    {task},
                    timeout=self.AVAILABILITY_CLEANUP_GRACE_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            raise

    async def _get_availability_access_token(
        self,
        client: httpx.AsyncClient,
        permit: _AvailabilityPermit,
    ) -> str:
        """Get a bounded token while preserving the existing shared cache."""
        if self._access_token and self._token_expires_at:
            if datetime.now(timezone.utc) < self._token_expires_at:
                return self._access_token

        token_url = (
            "https://login.microsoftonline.com/"
            + quote(self.tenant_id, safe="")
            + "/oauth2/v2.0/token"
        )
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }
        body = await self._run_availability_io(
            self._read_streamed_availability_response(
                client,
                "POST",
                token_url,
                _TOKEN_RESPONSE_LIMIT,
                token_boundary=True,
                data=data,
            ),
            permit,
        )
        try:
            token_data = json.loads(body, parse_constant=_reject_json_constant)
            token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
        except (AttributeError, KeyError, TypeError, ValueError):
            raise _AvailabilityReadFailure(
                AvailabilityFailureCategory.INVALID_RESPONSE,
                200,
            ) from None
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(expires_in, (int, float))
            or isinstance(expires_in, bool)
            or expires_in <= 0
        ):
            raise _AvailabilityReadFailure(
                AvailabilityFailureCategory.INVALID_RESPONSE,
                200,
            )
        try:
            lifetime = float(expires_in)
        except (OverflowError, TypeError, ValueError):
            raise _AvailabilityReadFailure(
                AvailabilityFailureCategory.INVALID_RESPONSE,
                200,
            ) from None
        if not math.isfinite(lifetime):
            raise _AvailabilityReadFailure(
                AvailabilityFailureCategory.INVALID_RESPONSE,
                200,
            )
        cache_seconds = min(lifetime, 3600.0) - 60.0
        if cache_seconds > 0:
            self._access_token = token
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=cache_seconds
            )
        return token

    @staticmethod
    def _availability_failure_result(
        query: AvailabilityQuery,
        category: AvailabilityFailureCategory,
        http_status: int | None = None,
    ) -> AvailabilityResult:
        if category is AvailabilityFailureCategory.INVALID_RESPONSE:
            status = AvailabilityStatus.INVALID_RESPONSE
        elif category is AvailabilityFailureCategory.INCOMPLETE:
            status = AvailabilityStatus.INCOMPLETE
        else:
            status = AvailabilityStatus.UNAVAILABLE
        return AvailabilityResult(
            query,
            status,
            datetime.now(timezone.utc),
            failure=AvailabilityFailure(category, http_status),
        )

    async def get_availability(
        self,
        query: AvailabilityQuery,
    ) -> AvailabilityResult:
        """Read one scoped Microsoft availability result without policy inference."""
        if type(query) is not AvailabilityQuery:
            raise AvailabilityContractError(_CONTRACT_ERROR)
        if (
            (self.tenant_id and query.scope.tenant_id != self.tenant_id)
            or (self.business_id and query.scope.business_id != self.business_id)
        ):
            raise AvailabilityContractError(_CONTRACT_ERROR)
        if not (
            self.tenant_id
            and self.client_id
            and self.client_secret
            and self.business_id
        ):
            return self._availability_failure_result(
                query,
                AvailabilityFailureCategory.NOT_CONFIGURED,
            )

        payload = {
            "staffIds": list(query.staff_ids),
            "startDateTime": {
                "dateTime": self._availability_wire_datetime(query.window.start),
                "timeZone": "UTC",
            },
            "endDateTime": {
                "dateTime": self._availability_wire_datetime(query.window.end),
                "timeZone": "UTC",
            },
        }
        url = (
            "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/"
            + quote(self.business_id, safe="")
            + "/getStaffAvailability"
        )
        permit = _AvailabilityPermit()
        try:
            async with asyncio.timeout(self.AVAILABILITY_TIMEOUT_SECONDS):
                await self._availability_slots.acquire()
                permit.acquired = True
                client = await self._get_client()
                token = await self._get_availability_access_token(client, permit)
                body = await self._run_availability_io(
                    self._read_streamed_availability_response(
                        client,
                        "POST",
                        url,
                        _AVAILABILITY_RESPONSE_LIMIT,
                        headers={
                            "Authorization": "Bearer " + token,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ),
                    permit,
                )
                try:
                    wire_payload = json.loads(body)
                except (TypeError, ValueError):
                    raise _AvailabilityReadFailure(
                        AvailabilityFailureCategory.INVALID_RESPONSE,
                        200,
                    ) from None
                observed_at = datetime.now(timezone.utc)
                return decode_availability(wire_payload, query, observed_at)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._availability_failure_result(
                query,
                AvailabilityFailureCategory.TIMEOUT,
            )
        except httpx.TimeoutException:
            return self._availability_failure_result(
                query,
                AvailabilityFailureCategory.TIMEOUT,
            )
        except httpx.TransportError:
            return self._availability_failure_result(
                query,
                AvailabilityFailureCategory.TRANSPORT,
            )
        except _AvailabilityReadFailure as error:
            return self._availability_failure_result(
                query,
                error.category,
                error.http_status,
            )
        except Exception:
            return self._availability_failure_result(
                query,
                AvailabilityFailureCategory.PROVIDER_ERROR,
            )
        finally:
            if permit.acquired and not permit.handed_off:
                self._availability_slots.release()

    async def get_service_facts(self, scope: CalendarScope) -> ServiceFactsRead:
        """Read exact service facts under the shared bounded availability budget."""
        if type(scope) is not CalendarScope:
            raise AvailabilityContractError(_CONTRACT_ERROR)
        if (scope.tenant_id != self.tenant_id
                or scope.business_id != self.business_id):
            raise AvailabilityContractError(_CONTRACT_ERROR)

        def failed(category: str) -> ServiceFactsRead:
            return ServiceFactsRead("unverified", scope, datetime.now(timezone.utc),
                                    failure=category)

        if not (self.tenant_id and self.client_id and self.client_secret
                and self.business_id):
            return failed("not_configured")
        url = (
            "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/"
            + quote(scope.business_id, safe="") + "/services/"
            + quote(scope.service_id, safe="")
        )
        permit = _AvailabilityPermit()
        try:
            async with asyncio.timeout(self.AVAILABILITY_TIMEOUT_SECONDS):
                await self._availability_slots.acquire()
                permit.acquired = True
                client = await self._get_client()
                token = await self._get_availability_access_token(client, permit)
                body = await self._run_availability_io(
                    self._read_streamed_availability_response(
                        client, "GET", url, _AVAILABILITY_RESPONSE_LIMIT,
                        headers={"Authorization": "Bearer " + token},
                    ), permit,
                )
                try:
                    wire = json.loads(body, parse_constant=_reject_json_constant)
                except (TypeError, ValueError):
                    return failed("invalid_response")
                return decode_service_facts(wire, scope, datetime.now(timezone.utc))
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return failed("timeout")
        except httpx.TimeoutException:
            return failed("timeout")
        except httpx.TransportError:
            return failed("transport")
        except _AvailabilityReadFailure as error:
            return failed(error.category.value)
        except Exception:
            return failed("provider_error")
        finally:
            if permit.acquired and not permit.handed_off:
                self._availability_slots.release()

    @staticmethod
    def _availability_wire_datetime(value: datetime) -> str:
        value = value.astimezone(timezone.utc)
        timespec = "microseconds" if value.microsecond else "seconds"
        return value.isoformat(timespec=timespec).replace("+00:00", "Z")

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
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
                if response.status_code == 204:
                    return {"deleted": True}
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

            # CRITICAL: The server runs in UTC. We MUST compute the availability
            # window using Eastern wall-clock time, because we label the window as
            # "Eastern Standard Time" to MS Graph. Using a naive datetime.now()
            # (UTC) here labeled as Eastern shifted the window ~4-5 hours into the
            # future, which excluded same-day morning/noon slots — callers could
            # not book "today" even when slots were free.
            now_eastern = business_now()
            # Small lead buffer so we don't offer a slot that is essentially "now".
            # Kept short (10 min) so near-term same-day bookings still work.
            start_date = now_eastern + timedelta(minutes=10)
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
                                start_dt = parse_graph_datetime(start_time_str)
                                end_dt = parse_graph_datetime(end_time_str)

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

    async def get_customer_appointments(self, customer_phone: str) -> List[Dict[str, Any]]:
        """
        Get upcoming appointments for a customer by phone number.

        Args:
            customer_phone: Customer's phone number

        Returns:
            List of appointment dicts with id, staff_name, start_time, service_name
        """
        normalized_phone = normalize_customer_phone(customer_phone)
        if not normalized_phone:
            logger.warning("MS Bookings: Invalid customer phone for appointment lookup")
            return []

        if not await self.is_available():
            logger.warning("MS Bookings: Not configured")
            return []

        try:
            # Get all appointments and filter by phone number
            # MS Graph doesn't support direct filtering by customerPhone, so we get recent appointments
            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/appointments"

            # Filter for future appointments
            now = business_now()
            params = {
                "$filter": f"startDateTime/dateTime ge '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
                "$orderby": "startDateTime/dateTime",
                "$top": 50
            }

            result = await self._make_request("GET", endpoint, params=params)

            if not result or 'value' not in result:
                logger.info("MS Bookings: No appointments found")
                return []

            appointments = []
            for appt in result['value']:
                normalized_appt_phone = normalize_customer_phone(
                    appointment_customer_phone(appt)
                )

                if normalized_appt_phone == normalized_phone:
                    # Parse start time - API returns in timezone specified when booking (Eastern)
                    # but we need to handle both UTC (Z suffix) and offset formats
                    start_dt_str = appt.get('startDateTime', {}).get('dateTime', '')
                    start_dt = None
                    formatted_time = "Unknown time"

                    if start_dt_str:
                        try:
                            # Parse ISO timestamp from MS Graph. It may arrive as:
                            #   - UTC with Z suffix:      2026-06-05T14:00:00Z
                            #   - With explicit offset:   2026-06-05T10:00:00-04:00
                            #   - Naive local (Eastern):  2026-06-05T10:00:00
                            # The old code subtracted a hardcoded 5 hours for the
                            # Z case — wrong during daylight saving (Eastern is
                            # UTC-4 in summer), shifting appointments by an hour
                            # and breaking the duplicate-day comparison. Use a
                            # DST-aware conversion instead.
                            start_dt = parse_graph_datetime(start_dt_str)
                            formatted_time = start_dt.strftime('%A, %B %d at %I:%M %p')
                        except Exception as e:
                            logger.warning(f"MS Bookings: Failed to parse time {start_dt_str}: {e}")

                    # Get staff name
                    staff_ids = appt.get('staffMemberIds', [])
                    staff_name = "Staff Member"
                    if staff_ids and self._staff_cache:
                        for s in self._staff_cache.values():
                            if s.id in staff_ids:
                                staff_name = s.name
                                break

                    # Log the actual appointment ID for debugging
                    appt_id = appt.get('id', '')
                    logger.info(f"MS Bookings: Found appointment ID={appt_id[:20]}... time={formatted_time}")

                    appointments.append({
                        'id': appt_id,
                        'staff_name': staff_name,
                        'customer_name': appt.get('customerName', ''),
                        'start_time': start_dt,
                        'formatted_time': formatted_time,
                        'service_name': appt.get('serviceName', 'Appointment')
                    })

            logger.info(f"MS Bookings: Found {len(appointments)} appointments for phone {customer_phone[-4:]}")
            return appointments

        except Exception as e:
            logger.error(f"MS Bookings: Failed to get customer appointments: {e}")
            return []

    async def cancel_customer_appointment(
        self, appointment_id: str, customer_phone: str
    ) -> bool:
        """
        Cancel an appointment by ID.

        Args:
            appointment_id: The MS Bookings appointment ID
            customer_phone: Phone number of the caller requesting cancellation

        Returns:
            True if cancelled successfully
        """
        normalized_phone = normalize_customer_phone(customer_phone)
        if not normalized_phone:
            logger.warning("MS Bookings: Invalid customer phone for cancellation")
            return False

        if not await self.is_available():
            logger.warning("MS Bookings: Not configured")
            return False

        try:
            endpoint = f"/solutions/bookingBusinesses/{self.business_id}/appointments/{appointment_id}"
            appointment = await self._make_request("GET", endpoint)
            owner_phone = normalize_customer_phone(
                appointment_customer_phone(appointment or {})
            )
            if owner_phone != normalized_phone:
                logger.warning("MS Bookings: Appointment cancellation ownership check failed")
                return False

            result = await self._make_request("DELETE", endpoint)

            if result and result.get('deleted'):
                logger.info(f"MS Bookings: Cancelled appointment {appointment_id}")
                return True
            else:
                logger.error(f"MS Bookings: Failed to cancel appointment {appointment_id}")
                return False

        except Exception as e:
            logger.error(f"MS Bookings: Failed to cancel appointment: {e}")
            return False

    async def close(self):
        """Close HTTP client"""
        pending_cleanup = tuple(self._availability_cleanup_tasks)
        if pending_cleanup:
            await asyncio.wait(
                pending_cleanup,
                timeout=self.AVAILABILITY_CLEANUP_GRACE_SECONDS,
            )
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
