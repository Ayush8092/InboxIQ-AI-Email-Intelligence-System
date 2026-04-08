"""
Feedback simulation and collection service.
Generates realistic feedback samples to bootstrap ML training.
Also provides the interface for capturing real user corrections.
"""
import random
from utils.secure_logger import get_secure_logger
from memory.repository import (
    insert_feedback, get_all_feedback, get_all_processed, get_all_emails,
)

logger = get_secure_logger(__name__)


# Simulated priority corrections based on email patterns
_SIMULATION_RULES = [
    # (subject_pattern, sender_pattern, correct_priority)
    ("server down",     None,              1),
    ("urgent",          None,              2),
    ("invoice",         "billing",         3),
    ("meeting",         None,              3),
    ("interview",       "recruit",         4),
    ("newsletter",      "newsletter",      7),
    ("shipped",         "amazon",          6),
    ("flash sale",      "deals",           7),
    ("password reset",  "security",        5),
    ("performance review", "hr",           3),
    ("security patch",  "cto",             1),
    ("budget",          None,              2),
    ("contract",        "legal",           2),
    ("feedback needed", None,              3),
    ("pull request",    None,              4),
]


def _match_rule(
    subject: str,
    sender: str,
    subject_pattern: str,
    sender_pattern: str | None,
) -> bool:
    """Check if email matches a simulation rule."""
    subj_match = subject_pattern.lower() in subject.lower()
    if not subj_match:
        return False
    if sender_pattern is None:
        return True
    return sender_pattern.lower() in sender.lower()


def simulate_feedback(
    n_samples: int = 20,
    dry_run: bool  = False,
) -> list[dict]:
    """
    Simulate realistic user feedback to bootstrap ML training.
    Uses rule-based simulation to generate priority corrections.

    n_samples: number of feedback samples to generate
    dry_run: if True, return samples without saving to DB

    Returns list of generated feedback dicts.
    """
    emails    = get_all_emails()
    processed = {p["email_id"]: p for p in get_all_processed()}

    if not emails:
        logger.warning("No emails to simulate feedback for")
        return []

    existing_feedback = {f["email_id"] for f in get_all_feedback()}
    generated         = []

    # Shuffle to avoid always picking same emails
    shuffled = list(emails)
    random.shuffle(shuffled)

    for email in shuffled:
        if len(generated) >= n_samples:
            break
        if email["id"] in existing_feedback:
            continue

        p            = processed.get(email["id"],{})
        current_pri  = p.get("priority", 5)
        subject      = email.get("subject","")
        sender       = email.get("sender","")
        correct_pri  = None

        # Apply simulation rules
        for subj_pat, send_pat, priority in _SIMULATION_RULES:
            if _match_rule(subject, sender, subj_pat, send_pat):
                correct_pri = priority
                break

        # Skip if rule says same priority (no correction needed)
        if correct_pri is None or correct_pri == current_pri:
            continue

        # Only correct if meaningful difference
        if abs(correct_pri - current_pri) < 1:
            continue

        fb = {
            "email_id":  email["id"],
            "field":     "priority",
            "old_value": str(current_pri),
            "new_value": str(correct_pri),
        }
        generated.append(fb)

        if not dry_run:
            insert_feedback(
                email["id"], "priority", str(current_pri), str(correct_pri)
            )
            logger.info(
                f"Simulated feedback | {email['id']} "
                f"priority {current_pri}→{correct_pri} | "
                f"subject='{subject[:40]}'"
            )

    logger.info(f"Generated {len(generated)} feedback samples (dry_run={dry_run})")
    return generated


def get_feedback_stats() -> dict:
    """Return statistics about the current feedback dataset."""
    feedback  = get_all_feedback()
    priority_fb = [f for f in feedback if f.get("field") == "priority"]
    category_fb = [f for f in feedback if f.get("field") == "category"]
    task_fb     = [f for f in feedback if f.get("field") == "task"]

    corrections_by_email: dict[str, int] = {}
    for fb in feedback:
        eid = fb.get("email_id","")
        corrections_by_email[eid] = corrections_by_email.get(eid, 0) + 1

    return {
        "total":            len(feedback),
        "priority":         len(priority_fb),
        "category":         len(category_fb),
        "task":             len(task_fb),
        "emails_corrected": len(corrections_by_email),
        "ready_for_ml":     len(priority_fb) >= 15,
    }