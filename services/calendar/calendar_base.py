"""
=====================================================
AI Voice Platform v2 - Calendar Service Interface
=====================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StaffMember:
    """Staff member info"""
    id: str
    name: str
    email: str
    role: str = ""


@dataclass
class Service:
    """Service info"""
    id: str
    name: str
    description: str
    duration_minutes: int
    price: Optional[float] = None


@dataclass
class TimeSlot:
    """Available time slot"""
    staff_id: str
    staff_name: str
    start_time: datetime
    end_time: datetime
    formatted: str


@dataclass
class BookingResult:
    """Booking result"""
    success: bool
    appointment_id: Optional[str] = None
    start_time: Optional[str] = None
    staff_name: Optional[str] = None
    customer_name: Optional[str] = None
    error_message: Optional[str] = None


class CalendarServiceBase(ABC):
    """
    Abstract base for calendar/booking services

    Provides appointment booking and availability checking.
    """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the service is configured and available"""
        pass

    @abstractmethod
    async def get_staff_members(self) -> List[StaffMember]:
        """Get all staff members"""
        pass

    @abstractmethod
    async def get_staff_by_name(self, name: str) -> Optional[StaffMember]:
        """Get staff member by name"""
        pass

    @abstractmethod
    async def get_services(self) -> List[Service]:
        """Get all services"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
