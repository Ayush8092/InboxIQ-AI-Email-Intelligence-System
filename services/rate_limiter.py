"""
Rate limiting service.
DB-backed per-user request throttling.
Protects against LLM overconsumption and Gmail quota exhaustion.
"""
from utils.secure_logger import get_secure_logger
from memory.repository import check_rate_limit

logger = get_secure_logger(__name__)

# Rate limit configs: {action: (max_count, window_minutes)}
RATE_LIMITS = {
    "process_emails":  (50,  60),    # 50 emails per hour
    "llm_call":        (200, 60),    # 200 LLM calls per hour
    "gmail_fetch":     (10,  60),    # 10 Gmail fetches per hour
    "train_model":     (5,   60),    # 5 model trainings per hour
    "generate_reply":  (30,  60),    # 30 reply drafts per hour
}

DEMO_RATE_LIMITS = {
    "process_emails":  (20,  60),
    "llm_call":        (100, 60),
    "gmail_fetch":     (3,   60),
    "train_model":     (2,   60),
    "generate_reply":  (15,  60),
}


def check_allowed(
    user_id: str,
    action: str,
    is_demo: bool = True,
) -> tuple[bool, str]:
    """
    Check if user is allowed to perform action.
    Returns (allowed: bool, message: str).
    """
    limits = DEMO_RATE_LIMITS if is_demo else RATE_LIMITS
    if action not in limits:
        return True, ""

    max_count, window_minutes = limits[action]
    allowed, remaining        = check_rate_limit(
        user_id, action, max_count, window_minutes
    )

    if not allowed:
        msg = (
            f"Rate limit reached for '{action}'. "
            f"Max {max_count} per {window_minutes} minutes. "
            f"Please wait before retrying."
        )
        logger.warning(f"Rate limit exceeded | user={user_id} action={action}")
        return False, msg

    if remaining <= 5:
        logger.info(f"Rate limit warning | user={user_id} action={action} remaining={remaining}")

    return True, f"{remaining} requests remaining"


def get_user_id(session_state: dict) -> str:
    """Get stable user identifier for rate limiting."""
    # Use email if authenticated, else "demo"
    email = session_state.get("user_email","")
    return email if email else "demo_user"