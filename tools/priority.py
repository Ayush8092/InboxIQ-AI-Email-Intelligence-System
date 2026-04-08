"""
Priority computation — hybrid rule + LLM.
Uses weighted multi-factor scoring instead of simple category lookup.
"""
from datetime import datetime, date
from config import SENDER_IMPORTANCE
from config.constants import (
    CATEGORY_PRIORITY_MAP,
    URGENCY_KEYWORDS_HIGH, URGENCY_KEYWORDS_MEDIUM,
)
from memory.repository import get_email, get_processed
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _deadline_score(deadline_str) -> int:
    """Returns 0-3 urgency from deadline. 0 = most urgent."""
    if not deadline_str:
        return 2
    try:
        dl    = datetime.fromisoformat(str(deadline_str)).date()
        days  = (dl - date.today()).days
        if days < 0:   return 0   # overdue
        if days <= 1:  return 0   # due today/tomorrow
        if days <= 3:  return 1
        if days <= 7:  return 2
        return 3
    except Exception:
        return 2


def _keyword_urgency(text: str) -> int:
    """Returns 0-2 urgency from keywords. 0 = most urgent."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in URGENCY_KEYWORDS_HIGH):
        return 0
    if any(kw in text_lower for kw in URGENCY_KEYWORDS_MEDIUM):
        return 1
    return 2


def _sender_urgency(sender: str) -> int:
    """Returns 0-2 urgency from sender. 0 = most urgent."""
    imp = SENDER_IMPORTANCE.get(sender, 0)
    if imp >= 9:  return 0
    if imp >= 7:  return 1
    return 2


def compute_priority(email_id: str) -> dict:
    """
    Weighted multi-factor priority scoring:
    - Category base priority (40% weight)
    - Keyword urgency (30% weight)
    - Sender importance (20% weight)
    - Deadline proximity (10% weight)

    Result is 1-7 scale with hard overrides for critical signals.
    """
    email     = get_email(email_id)
    processed = get_processed(email_id) or {}
    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    text     = email["subject"] + " " + email["body"]
    sender   = email["sender"]
    category = processed.get("category", "General Info") or "General Info"
    deadline = processed.get("deadline")

    # ── Factor scores (all 0-6 scale, 0=most urgent) ─────────────────────────
    cat_base     = CATEGORY_PRIORITY_MAP.get(category, 5) - 1        # 0-6
    kw_urgency   = _keyword_urgency(text) * 2                         # 0, 2, 4
    sender_urg   = _sender_urgency(sender) * 2                        # 0, 2, 4
    deadline_urg = _deadline_score(deadline) * 1.5                    # 0, 1.5, 3, 4.5

    # ── Weighted average ──────────────────────────────────────────────────────
    raw_score = (
        cat_base     * 0.40 +
        kw_urgency   * 0.30 +
        sender_urg   * 0.20 +
        deadline_urg * 0.10
    )

    priority = max(1, min(7, round(raw_score + 1)))

    # ── Hard overrides — domain rules that cannot be overridden ───────────────
    text_lower = text.lower()

    # Server/production down → always Critical
    if any(kw in text_lower for kw in ["server is down","production down","outage","503 error","system failure","not responding"]):
        priority = 1
        logger.info(f"Priority override → Critical (production alert) | {email_id}")

    # Security/CVE → always High or Critical
    elif any(kw in text_lower for kw in ["critical cve","security breach","security patch","vulnerability","deploy by eod"]):
        priority = min(priority, 2)
        logger.info(f"Priority override → High (security) | {email_id}")

    # Boss/CTO + urgent → Critical
    elif SENDER_IMPORTANCE.get(sender, 0) >= 10 and _keyword_urgency(text) == 0:
        priority = 1
        logger.info(f"Priority override → Critical (senior sender + urgent) | {email_id}")

    # Newsletter/Social → never above Low
    elif category in {"Newsletter", "Social / Notification"}:
        priority = max(priority, 6)

    # Overdue deadline → boost by 2 levels
    if deadline and _deadline_score(deadline) == 0 and priority > 2:
        priority = max(1, priority - 2)
        logger.info(f"Priority boosted (overdue deadline) | {email_id}")

    logger.debug(
        f"compute_priority | {email_id} → {priority} "
        f"(cat={cat_base:.1f} kw={kw_urgency} sender={sender_urg} dl={deadline_urg:.1f})"
    )
    return {"priority": priority}