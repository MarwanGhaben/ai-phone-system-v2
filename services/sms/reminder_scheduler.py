"""
=====================================================
AI Voice Platform v2 - Appointment Reminder Scheduler
=====================================================

Schedules and sends SMS reminders 24 hours before appointments.
Persists pending reminders to disk so they survive restarts.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from loguru import logger


class ReminderScheduler:
    """
    Background scheduler that sends SMS reminders 24 hours before appointments.

    - Stores reminders in a JSON file (data/reminders.json)
    - Checks every 60 seconds for reminders due
    - Marks reminders as sent so they aren't repeated
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            storage_path = str(data_dir / "reminders.json")

        self.storage_path = storage_path
        self._reminders = []
        self._load()
        self._task: Optional[asyncio.Task] = None

    def _load(self):
        """Load reminders from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._reminders = json.load(f)
                logger.info(f"Reminder scheduler: Loaded {len(self._reminders)} pending reminders")
        except Exception as e:
            logger.error(f"Reminder scheduler: Failed to load: {e}")
            self._reminders = []

    def _save(self):
        """Save reminders to disk"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._reminders, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Reminder scheduler: Failed to save: {e}")

    def schedule_reminder(
        self,
        phone_number: str,
        customer_name: str,
        staff_name: str,
        appointment_time_str: str,
        appointment_dt: datetime,
        language: str = "en",
    ) -> None:
        """
        Schedule a reminder to be sent 24 hours before the appointment.

        Args:
            phone_number: Caller's phone number
            customer_name: Customer name
            staff_name: Staff/accountant name
            appointment_time_str: Human-readable appointment time
            appointment_dt: Appointment datetime object
            language: 'en' or 'ar'
        """
        remind_at = appointment_dt - timedelta(hours=24)

        # Don't schedule if the reminder time has already passed
        if remind_at <= datetime.now():
            logger.info(f"Reminder scheduler: Appointment too soon for 24hr reminder ({appointment_time_str}), skipping")
            return

        reminder = {
            "phone_number": phone_number,
            "customer_name": customer_name,
            "staff_name": staff_name,
            "appointment_time_str": appointment_time_str,
            "appointment_dt": appointment_dt.isoformat(),
            "remind_at": remind_at.isoformat(),
            "language": language,
            "sent": False,
            "created_at": datetime.now().isoformat(),
        }

        # Cancel any existing unsent reminders for this phone number first
        # This ensures rebooking replaces old reminders instead of duplicating
        self.cancel_reminders_for_phone(phone_number)

        self._reminders.append(reminder)
        self._save()
        logger.info(f"Reminder scheduler: Scheduled for {phone_number} at {remind_at.isoformat()} (appointment: {appointment_time_str})")

    def cancel_reminders_for_phone(self, phone_number: str) -> int:
        """
        Cancel all pending (unsent) reminders for a specific phone number.

        This is called when:
        - An appointment is cancelled
        - A new appointment is booked (replaces old reminder)

        Args:
            phone_number: The phone number to cancel reminders for

        Returns:
            Number of reminders cancelled
        """
        original_count = len(self._reminders)

        # Remove all unsent reminders for this phone number
        self._reminders = [
            r for r in self._reminders
            if r.get("sent") or r.get("phone_number") != phone_number
        ]

        cancelled_count = original_count - len(self._reminders)

        if cancelled_count > 0:
            self._save()
            logger.info(f"Reminder scheduler: Cancelled {cancelled_count} pending reminder(s) for {phone_number}")

        return cancelled_count

    def start(self) -> None:
        """Start the background checker loop"""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Reminder scheduler: Background loop started")

    def stop(self) -> None:
        """Stop the background loop"""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Reminder scheduler: Background loop stopped")

    async def _check_loop(self):
        """Check for due reminders every 60 seconds"""
        while True:
            try:
                await self._process_due_reminders()
            except Exception as e:
                logger.error(f"Reminder scheduler: Error in check loop: {e}")

            await asyncio.sleep(60)

    async def _process_due_reminders(self):
        """Find and send any reminders that are due"""
        now = datetime.now()
        sent_count = 0

        for reminder in self._reminders:
            if reminder.get("sent"):
                continue

            remind_at = datetime.fromisoformat(reminder["remind_at"])
            if remind_at <= now:
                # This reminder is due — send it
                success = await self._send_reminder(reminder)
                if success:
                    reminder["sent"] = True
                    reminder["sent_at"] = now.isoformat()
                    sent_count += 1

        if sent_count > 0:
            self._save()
            logger.info(f"Reminder scheduler: Sent {sent_count} reminder(s)")

        # Clean up old reminders (sent + appointment passed)
        self._reminders = [
            r for r in self._reminders
            if not r.get("sent") or datetime.fromisoformat(r["appointment_dt"]) > now
        ]

    async def _send_reminder(self, reminder: dict) -> bool:
        """Send a single reminder SMS"""
        from services.sms.telnyx_sms_service import get_sms_service

        sms = get_sms_service()
        if not sms.is_available():
            logger.warning("Reminder scheduler: SMS not configured, cannot send reminder")
            return False

        try:
            sent = await sms.send_appointment_reminder(
                to_number=reminder["phone_number"],
                customer_name=reminder["customer_name"],
                staff_name=reminder["staff_name"],
                appointment_time=reminder["appointment_time_str"],
                language=reminder.get("language", "en"),
            )
            if sent:
                logger.info(f"Reminder scheduler: Sent reminder to {reminder['phone_number']}")
            return sent
        except Exception as e:
            logger.error(f"Reminder scheduler: Failed to send: {e}")
            return False


# Global instance
_scheduler: Optional[ReminderScheduler] = None


def get_reminder_scheduler() -> ReminderScheduler:
    """Get global reminder scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = ReminderScheduler()
    return _scheduler
