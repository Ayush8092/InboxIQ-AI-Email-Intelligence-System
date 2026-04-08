"""
Email processing service — business logic decoupled from UI.
Orchestrates the full email pipeline.
"""
from utils.secure_logger import get_secure_logger
from memory.repository import (
    get_all_emails, get_all_processed, get_processed,
    insert_email,
)
from agent.orchestrator import orchestrator

logger = get_secure_logger(__name__)


def process_email(
    email_id: str,
    persona: str   = "Formal",
    dry_run: bool  = False,
    user_command: str = "Handle this email",
) -> dict:
    """
    Process a single email through the full pipeline.
    Returns result dict — never raises.
    """
    try:
        result = orchestrator.handle_email(
            email_id,
            persona=persona,
            dry_run=dry_run,
            user_command=user_command,
        )
        logger.info(
            f"Processed email | id={email_id} "
            f"category={result.get('final_state',{}).get('category')} "
            f"priority={result.get('final_state',{}).get('priority')}"
        )
        return result
    except Exception as e:
        logger.error(f"Email processing failed | id={email_id} error={e}")
        return {"error": str(e), "email_id": email_id}


def process_all_emails(
    emails: list[dict],
    persona: str  = "Formal",
    dry_run: bool = False,
) -> list[dict]:
    """
    Process all emails sequentially.
    Returns list of results.
    """
    results = []
    for email in emails:
        result = process_email(
            email["id"],
            persona=persona,
            dry_run=dry_run,
        )
        results.append(result)
    return results


def import_gmail_emails(access_token: str, max_results: int = 20) -> list[dict]:
    """
    Fetch and import Gmail emails into local DB.
    Returns list of imported email dicts.
    """
    from utils.oauth import fetch_gmail_emails
    emails = fetch_gmail_emails(access_token, max_results=max_results)
    for email in emails:
        try:
            insert_email(email)
        except Exception as e:
            logger.warning(f"Failed to insert email {email.get('id')}: {e}")
    logger.info(f"Imported {len(emails)} Gmail emails")
    return emails