"""
Email summarization with:
- HTML cleaning BEFORE LLM (fixes empty summary bug)
- Rule-based fallback (never returns empty)
- Cached LLM calls
"""
from utils.llm_client import call_llm
from utils.email_cleaner import clean_email_body
from utils.secure_logger import get_secure_logger
from memory.repository import get_email, get_prompt

logger = get_secure_logger(__name__)

_SUMMARIZE_PROMPT = """\
Summarize this email in 1-2 sentences. Be specific and factual.

Subject: {subject}
From: {sender}
Body: {body}

Reply with ONLY the summary sentence(s), no preamble."""


def _rule_based_summary(email: dict) -> str:
    """
    Generate summary without LLM.
    Used when LLM fails or body is empty after cleaning.
    """
    subject = email.get("subject", "No subject")
    sender  = email.get("sender", "Unknown")
    body    = email.get("body", "")
    clean   = clean_email_body(body, max_chars=400)

    # Get first meaningful sentence
    sentences = [
        s.strip()
        for s in clean.replace("\n", " ").split(".")
        if len(s.strip()) > 25
    ]
    if sentences:
        first = sentences[0].strip()
        if len(first) > 160:
            first = first[:157] + "..."
        return f"{subject} — {first}."
    return f"Email from {sender}: {subject}"


def summarize_email(email_id: str) -> dict:
    """
    Summarize with cleaned body.
    Falls back to rule-based if LLM fails or returns empty.
    """
    email = get_email(email_id)
    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    # Clean body BEFORE LLM — this fixes the empty summary bug
    clean_body = clean_email_body(email.get("body", ""), max_chars=600)

    # Skip LLM if body is too short after cleaning
    if len(clean_body) < 30:
        summary = _rule_based_summary(email)
        logger.debug(f"summarize:rule_short | {email_id}")
        return {"summary": summary}

    # Build prompt with clean text
    template = get_prompt("summarize") or _SUMMARIZE_PROMPT
    try:
        prompt = template.format(
            subject=email.get("subject", ""),
            body=clean_body,
            sender=email.get("sender", ""),
        )
    except KeyError:
        prompt = _SUMMARIZE_PROMPT.format(
            subject=email.get("subject", ""),
            body=clean_body,
            sender=email.get("sender", ""),
        )

    raw     = call_llm(prompt, temperature=0.1, max_tokens=120, use_cache=True)
    summary = raw.strip() if raw and len(raw.strip()) > 10 else None

    if not summary:
        summary = _rule_based_summary(email)
        logger.warning(f"summarize:llm_empty | {email_id} → using rule-based")
    else:
        logger.debug(f"summarize:llm | {email_id}")

    return {"summary": summary}
