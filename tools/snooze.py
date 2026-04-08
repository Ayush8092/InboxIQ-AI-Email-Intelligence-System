"""
Intelligent email snooze.
Determines snooze duration based on urgency, category, and deadline.
"""
from datetime import datetime, timedelta
from config.constants import (
    SNOOZE_DEFAULT_HOURS,
    SNOOZE_LOW_PRIORITY_HOURS,
    SNOOZE_NEWSLETTER_HOURS,
)
from memory.repository import get_email, get_processed, update_processed_field
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _smart_snooze_hours(email: dict, processed: dict) -> tuple[int, str]:
    """
    Determine snooze duration intelligently.

    Rules:
    - Overdue or urgent deadlines → refuse snooze (return 0)
    - Newsletter/Social → long snooze (72h)
    - Low priority (6-7) → medium snooze (48h)
    - Default → standard snooze (24h)

    Returns (hours, reason)
    """
    category = processed.get("category", "")
    priority = processed.get("priority", 5)
    deadline = processed.get("deadline")

    # Check if deadline is imminent — refuse snooze
    if deadline:
        try:
            dl   = datetime.fromisoformat(str(deadline)).date()
            days = (dl - datetime.utcnow().date()).days
            if days <= 1:
                return 0, f"Cannot snooze — deadline is {'today' if days == 0 else 'overdue'}"
            if days <= 3:
                return 8, f"Short snooze only — deadline in {days} days"
        except Exception:
            pass

    # Critical/High priority — refuse snooze
    if priority <= 2:
        return 0, f"Cannot snooze — priority is {'Critical' if priority == 1 else 'High'}"

    # Newsletter/Social → long snooze
    if category in {"Newsletter", "Social / Notification"}:
        return SNOOZE_NEWSLETTER_HOURS, f"{SNOOZE_NEWSLETTER_HOURS}h snooze (low-value category)"

    # Low priority → medium snooze
    if priority >= 6:
        return SNOOZE_LOW_PRIORITY_HOURS, f"{SNOOZE_LOW_PRIORITY_HOURS}h snooze (low priority)"

    # Default
    return SNOOZE_DEFAULT_HOURS, f"{SNOOZE_DEFAULT_HOURS}h standard snooze"


def snooze_email(email_id: str, hours: int = None) -> dict:
    """
    Intelligently snooze an email.
    If hours is None, determines duration automatically based on context.
    """
    email     = get_email(email_id)
    processed = get_processed(email_id) or {}

    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    # Use intelligent snooze if hours not explicitly specified
    if hours is None:
        smart_hours, reason = _smart_snooze_hours(email, processed)
    else:
        smart_hours = hours
        reason      = f"Manual snooze for {hours}h"

    if smart_hours == 0:
        logger.info(f"snooze refused | {email_id} — {reason}")
        return {
            "email_id":      email_id,
            "snoozed":       False,
            "reason":        reason,
            "snoozed_until": None,
            "hours":         0,
        }

    snooze_until = (datetime.utcnow() + timedelta(hours=smart_hours)).isoformat()
    update_processed_field(email_id, "snoozed_until", snooze_until)
    logger.info(f"snooze | {email_id} until {snooze_until} ({reason})")

    return {
        "email_id":      email_id,
        "snoozed":       True,
        "reason":        reason,
        "snoozed_until": snooze_until,
        "hours":         smart_hours,
    }