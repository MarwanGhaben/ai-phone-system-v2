"""
=====================================================
AI Voice Platform v2 - Telnyx SMS Service
=====================================================

Sends SMS via Telnyx API for:
- Booking confirmations (immediately after booking)
- Appointment reminders (24 hours before)
"""

import httpx
from typing import Optional
from loguru import logger
from config.settings import get_settings


class TelnyxSMSService:
    """
    Send SMS messages via Telnyx REST API.
    Uses httpx directly (no SDK needed).
    """

    API_URL = "https://api.telnyx.com/v2/messages"

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.telnyx_api_key
        self.from_number = settings.telnyx_phone_number
        self._available = bool(self.api_key and self.from_number)

        if self._available:
            logger.info(f"Telnyx SMS: Configured (from={self.from_number})")
        else:
            logger.warning("Telnyx SMS: Not configured (missing TELNYX_API_KEY or TELNYX_PHONE_NUMBER)")

    def is_available(self) -> bool:
        return self._available

    async def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send an SMS message.

        Args:
            to_number: Recipient phone number (E.164 format, e.g. +16471234567)
            message: Message text

        Returns:
            True if sent successfully
        """
        if not self._available:
            logger.warning("Telnyx SMS: Cannot send - not configured")
            return False

        if not to_number or to_number in ["+0000000000", "unknown", "private"]:
            logger.warning(f"Telnyx SMS: Cannot send - invalid number: {to_number}")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "from": self.from_number,
                        "to": to_number,
                        "text": message,
                    },
                    timeout=10.0,
                )

            if response.status_code in (200, 201, 202):
                data = response.json().get("data", {})
                msg_id = data.get("id", "unknown")
                logger.info(f"Telnyx SMS: Sent to {to_number} (id={msg_id})")
                return True
            else:
                logger.error(f"Telnyx SMS: Failed {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telnyx SMS: Exception sending to {to_number}: {e}")
            return False

    async def send_booking_confirmation(
        self,
        to_number: str,
        customer_name: str,
        staff_name: str,
        appointment_time: str,
        language: str = "en",
    ) -> bool:
        """
        Send booking confirmation SMS.

        Args:
            to_number: Caller's phone number
            customer_name: Customer name
            staff_name: Accountant/staff name
            appointment_time: Formatted appointment time string
            language: 'en' or 'ar'
        """
        if language == "ar":
            message = (
                f"مرحباً {customer_name}،\n"
                f"تم تأكيد موعدك مع {staff_name} في {appointment_time}.\n"
                f"شركة فليكسبل أكاونتنغ\n"
                f"للإلغاء أو التعديل يرجى الاتصال بنا."
            )
        else:
            message = (
                f"Hi {customer_name},\n"
                f"Your appointment with {staff_name} is confirmed for {appointment_time}.\n"
                f"Flexible Accounting\n"
                f"To cancel or reschedule, please call us."
            )

        logger.info(f"Telnyx SMS: Sending booking confirmation to {to_number}")
        return await self.send_sms(to_number, message)

    async def send_appointment_reminder(
        self,
        to_number: str,
        customer_name: str,
        staff_name: str,
        appointment_time: str,
        language: str = "en",
    ) -> bool:
        """
        Send 24-hour appointment reminder SMS.
        """
        if language == "ar":
            message = (
                f"تذكير: مرحباً {customer_name}،\n"
                f"لديك موعد غداً مع {staff_name} في {appointment_time}.\n"
                f"شركة فليكسبل أكاونتنغ\n"
                f"للإلغاء أو التعديل يرجى الاتصال بنا."
            )
        else:
            message = (
                f"Reminder: Hi {customer_name},\n"
                f"You have an appointment tomorrow with {staff_name} at {appointment_time}.\n"
                f"Flexible Accounting\n"
                f"To cancel or reschedule, please call us."
            )

        logger.info(f"Telnyx SMS: Sending reminder to {to_number}")
        return await self.send_sms(to_number, message)


# Global instance
_sms_service: Optional[TelnyxSMSService] = None


def get_sms_service() -> TelnyxSMSService:
    """Get global SMS service instance"""
    global _sms_service
    if _sms_service is None:
        _sms_service = TelnyxSMSService()
    return _sms_service
