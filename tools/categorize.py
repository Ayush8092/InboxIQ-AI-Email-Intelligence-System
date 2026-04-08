"""
Four-layer email categorization.

Architecture:
  Layer 0: Content-first keyword scoring (subject + body, NOT just sender)
  Layer 1: LLM with cleaned body (only when rules are ambiguous)
  Layer 2: LLM retry with simpler prompt
  Layer 3: Heuristic fallback (guaranteed result, never fails)

Root cause fix:
  OLD: sender-domain overrides blindly mapped glassdoor → Job always
  NEW: subject + body scored first, sender is ONE signal among many
       "How do people work for 50 years?" → Newsletter (body content wins)
       "Data Scientist at Siemens" → Job/Recruitment (subject wins)
"""
import re
from utils.llm_client import call_llm
from utils.validators import parse_json_strict, validate_category_output
from utils.email_cleaner import clean_email_body
from utils.secure_logger import get_secure_logger
from memory.repository import get_email, get_feedback_preferences
from config.constants import CATEGORIES, CONFIDENCE_CAP

logger = get_secure_logger(__name__)

# ── Category keyword maps (scored against CONTENT, not just sender) ───────────

_CATEGORY_KEYWORDS: dict[str, dict] = {
    "Job / Recruitment": {
        "subject_strong": [
            # Role titles in subject — very strong signal
            r"\b(software|data|ml|ai|backend|frontend|fullstack|full.stack|devops|"
            r"sde|sde-\d|engineer|developer|analyst|scientist|manager|architect|"
            r"intern|fresher|associate|lead|senior|junior)\b",
            r"\b(job alert|job match|recommended job|new job|opening|vacancy|"
            r"position|role|hiring|opportunity|apply now)\b",
            r"\b(interview|offer letter|job description|jd)\b",
        ],
        "subject_weak": [
            r"\b(career|resume|cv|skill|experience|ctc|salary|package)\b",
        ],
        "body_strong": [
            r"\b(apply now|apply here|view job|job description|required skills|"
            r"years of experience|ctc|lpa|per annum|work from|remote|hybrid|onsite)\b",
        ],
        "sender_boost": [
            "glassdoor", "naukri", "linkedin", "indeed", "monster", "shine",
            "foundit", "instahyre", "unstop", "internshala", "wellfound",
            "hirist", "cutshort", "apna", "resume.io", "alerts.resume",
            "jobs@", "career@", "hiring@", "recruit@",
        ],
        # These subjects from job senders should NOT be categorized as job
        "subject_override_block": [
            r"how do (people|you|professionals)",
            r"top \d+ (tips|ways|mistakes|skills)",
            r"newsletter|weekly digest|roundup",
            r"salary survey|industry report|trends in",
            r"congratulations|welcome to",
        ],
    },

    "Newsletter": {
        "subject_strong": [
            r"\b(newsletter|digest|weekly|monthly|roundup|edition|issue #)\b",
            r"\b(top \d+|this week|this month|trending)\b",
        ],
        "subject_weak": [
            r"\b(tips|guide|how to|why|what is|explained)\b",
        ],
        "body_strong": [
            r"unsubscribe|manage preferences|update email preferences",
            r"you(?:'re| are) receiving this because",
        ],
        "sender_boost": [
            "newsletter@", "digest@", "weekly@", "noreply@substack",
            "mailer@medium", "noreply@beehiiv", "hello@morning",
            "email@glassdoor",  # glassdoor career advice newsletters
        ],
        "subject_override_block": [],
    },

    "Alert / Urgent": {
        "subject_strong": [
            r"\b(server down|production down|outage|incident|503|500 error|"
            r"not responding|unreachable|failed|critical error|security breach)\b",
            r"\b(alert|urgent|immediate|asap|emergency)\b",
        ],
        "subject_weak": [
            r"\b(warning|error|issue|problem)\b",
        ],
        "body_strong": [
            r"\b(is down|went down|not available|high cpu|memory leak|"
            r"disk full|deployment failed)\b",
        ],
        "sender_boost": [
            "alerts@", "pagerduty", "opsgenie", "newrelic",
            "datadog", "cloudwatch", "nagios", "zabbix",
            "security@", "noreply@security",
        ],
        "subject_override_block": [],
    },

    "Action Required": {
        "subject_strong": [
            r"\b(action required|please (?:approve|review|confirm|sign|submit)|"
            r"approval needed|your response|response required)\b",
            r"\b(deadline|due (?:today|tomorrow|friday|monday)|by eod)\b",
        ],
        "subject_weak": [
            r"\b(feedback|review|confirm|approve)\b",
        ],
        "body_strong": [
            r"\b(please (?:review|approve|confirm|sign|complete)|"
            r"action required|your approval|sign by|must be done)\b",
        ],
        "sender_boost": [],
        "subject_override_block": [],
    },

    "Billing / Invoice": {
        "subject_strong": [
            r"\b(invoice|payment due|billing|subscription (?:renewal|expired)|"
            r"receipt|amount due|your (?:bill|statement)|charged)\b",
        ],
        "subject_weak": [
            r"\b(payment|subscription|renew|refund)\b",
        ],
        "body_strong": [
            r"\b(invoice number|amount|due date|pay now|payment method|"
            r"total amount|billing cycle)\b",
        ],
        "sender_boost": [
            "billing@", "invoice@", "payment@", "noreply@stripe",
            "receipts@", "accounts@", "noreply@notion",
            "no-reply@github", "paypal",
        ],
        "subject_override_block": [],
    },

    "Meeting / Event": {
        "subject_strong": [
            r"\b(meeting invite|calendar invite|you(?:'re| are) invited|"
            r"webinar|conference|event|zoom|google meet|teams call)\b",
            r"\b(rsvp|join us|register now|lunch|team (?:lunch|outing|event))\b",
        ],
        "subject_weak": [
            r"\b(meeting|call|sync|standup|catchup)\b",
        ],
        "body_strong": [
            r"\b(zoom link|meeting link|join the meeting|add to calendar|"
            r"google calendar|outlook calendar)\b",
        ],
        "sender_boost": [
            "calendar@", "events@", "noreply@zoom",
            "no-reply@meetup", "eventbrite",
        ],
        "subject_override_block": [],
    },

    "Travel": {
        "subject_strong": [
            r"\b(flight (?:confirmed|booked|cancelled)|booking confirmed|"
            r"pnr|itinerary|hotel (?:confirmed|booked)|check-in)\b",
        ],
        "subject_weak": [
            r"\b(flight|hotel|trip|travel|departure)\b",
        ],
        "body_strong": [
            r"\b(pnr number|gate number|boarding pass|seat number|"
            r"departure time|arrival time|check-in time)\b",
        ],
        "sender_boost": [
            "noreply@indigo", "noreply@airindia", "makemytrip",
            "cleartrip", "booking.com", "irctc", "goibibo",
        ],
        "subject_override_block": [],
    },

    "Social / Notification": {
        "subject_strong": [
            r"\b(your order|has shipped|out for delivery|delivered|"
            r"order (?:confirmed|shipped|delivered|#))\b",
            r"\b(new connection|connected with|accepted your|"
            r"liked your|commented on)\b",
        ],
        "subject_weak": [
            r"\b(notification|update|reminder)\b",
        ],
        "body_strong": [
            r"\b(track your order|tracking number|estimated delivery)\b",
        ],
        "sender_boost": [
            "noreply@amazon", "shipping@amazon", "auto-confirm@amazon",
            "noreply@flipkart", "noreply@twitter",
            "notification@facebook", "noreply@instagram",
        ],
        "subject_override_block": [],
    },
}

_CATEGORIES_LIST = list(_CATEGORY_KEYWORDS.keys()) + ["General Info"]


def _content_score(email: dict) -> tuple[str, float] | None:
    """
    Score ALL categories based on subject + body content.
    Returns (best_category, confidence) if confident enough, else None.
    This is content-first — sender is just one signal, not the decision.
    """
    subject = email.get("subject", "").lower()
    body    = clean_email_body(email.get("body", ""), max_chars=400).lower()
    sender  = email.get("sender", "").lower()
    text    = subject + " " + body

    scores: dict[str, float] = {cat: 0.0 for cat in _CATEGORY_KEYWORDS}

    for cat, rules in _CATEGORY_KEYWORDS.items():
        # Check override block FIRST — prevent sender bias
        blocked = False
        for block_pat in rules.get("subject_override_block", []):
            if re.search(block_pat, subject, re.IGNORECASE):
                blocked = True
                break
        if blocked:
            continue

        # Subject strong match — highest weight
        for pat in rules.get("subject_strong", []):
            if re.search(pat, subject, re.IGNORECASE):
                scores[cat] += 4.0

        # Subject weak match
        for pat in rules.get("subject_weak", []):
            if re.search(pat, subject, re.IGNORECASE):
                scores[cat] += 1.5

        # Body match
        for pat in rules.get("body_strong", []):
            if re.search(pat, body, re.IGNORECASE):
                scores[cat] += 2.0

        # Sender boost — only adds points, never wins alone
        for s_pattern in rules.get("sender_boost", []):
            if s_pattern in sender:
                scores[cat] += 1.5  # sender is worth 1.5, not a veto
                break

    # Find winner
    if not scores or max(scores.values()) == 0:
        return None

    sorted_cats  = sorted(scores.items(), key=lambda x: -x[1])
    best_cat, best_score = sorted_cats[0]
    second_score = sorted_cats[1][1] if len(sorted_cats) > 1 else 0.0

    # Only return if clear winner (not tied)
    if best_score == 0:
        return None

    # Confidence = separation between 1st and 2nd
    margin = best_score - second_score
    if margin < 1.0 and best_score < 4.0:
        # Too close to call without LLM
        return None

    # Map score to confidence
    conf = min(0.92, 0.55 + min(best_score, 12.0) / 20.0)
    return best_cat, round(conf, 2)


def _heuristic_fallback(email: dict) -> tuple[str, float]:
    """
    Pure keyword fallback — always returns a result.
    Used when both rule scoring and LLM fail.
    """
    result = _content_score(email)
    if result:
        return result
    return "General Info", 0.40


_LLM_PROMPT = """\
Classify this email into exactly ONE category.

Subject: {subject}
Body: {body}

Categories (choose exactly one):
{cats}

Rules:
- Read the SUBJECT and BODY carefully
- "How do people work..." type subjects → Newsletter
- Job title + company in subject → Job / Recruitment
- Server down / outage → Alert / Urgent
- Invoice / payment due → Billing / Invoice
- Meeting invite → Meeting / Event

Reply ONLY with JSON:
{{"category": "EXACT_CATEGORY_NAME", "confidence": 0.85}}"""

_NORM: dict[str, str] = {}
for _c in _CATEGORIES_LIST:
    _NORM[_c.lower().strip()] = _c
_NORM.update({
    "action required":       "Action Required",
    "meeting/event":         "Meeting / Event",
    "meeting / event":       "Meeting / Event",
    "billing/invoice":       "Billing / Invoice",
    "billing / invoice":     "Billing / Invoice",
    "alert/urgent":          "Alert / Urgent",
    "alert / urgent":        "Alert / Urgent",
    "social/notification":   "Social / Notification",
    "social / notification": "Social / Notification",
    "job/recruitment":       "Job / Recruitment",
    "job / recruitment":     "Job / Recruitment",
    "general info":          "General Info",
    "general information":   "General Info",
})


def _normalize(cat: str) -> str | None:
    if not cat:
        return None
    cat = cat.strip()
    if cat in _CATEGORIES_LIST:
        return cat
    found = _NORM.get(cat.lower().strip())
    if found:
        return found
    for c in _CATEGORIES_LIST:
        if cat.lower() in c.lower():
            return c
    return None


def _apply_feedback(
    email: dict, category: str, confidence: float, prefs: dict
) -> tuple[str, float]:
    sender   = email.get("sender", "")
    overrides = prefs.get("sender_category_overrides", {})
    if sender in overrides:
        return overrides[sender], 0.92
    corrections = prefs.get("category_corrections", {})
    if category in corrections:
        return corrections[category], min(confidence + 0.08, CONFIDENCE_CAP)
    return category, confidence


def categorize_email(email_id: str) -> dict:
    """
    Four-layer content-first categorization.

    Layer 0: Content scoring (subject+body keywords) — no LLM
    Layer 1: LLM with cleaned body when rules are ambiguous
    Layer 2: LLM retry simpler prompt
    Layer 3: Heuristic fallback — guaranteed result
    """
    email = get_email(email_id)
    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    prefs = get_feedback_preferences()

    # ── Layer 0: Content-first scoring ───────────────────────────────────────
    rule_result = _content_score(email)
    if rule_result:
        cat, conf = rule_result
        cat, conf = _apply_feedback(email, cat, conf, prefs)
        logger.debug(f"categorize:rule | {email_id} → {cat} {conf:.2f}")
        return {"category": cat, "confidence": conf}

    # ── Clean body for LLM ────────────────────────────────────────────────────
    clean_body = clean_email_body(email.get("body", ""), max_chars=500)
    cats_str   = "\n".join(f"- {c}" for c in _CATEGORIES_LIST)

    # ── Layer 1: LLM ─────────────────────────────────────────────────────────
    prompt1 = _LLM_PROMPT.format(
        subject=email.get("subject", ""),
        body=clean_body[:400],
        cats=cats_str,
    )
    raw1    = call_llm(prompt1, temperature=0.0, max_tokens=60, use_cache=True)
    result1 = parse_json_strict(raw1, fallback={}, context=f"cat1_{email_id}")

    if validate_category_output(result1):
        norm = _normalize(result1.get("category", ""))
        if norm:
            conf       = min(float(result1.get("confidence", 0.75)), CONFIDENCE_CAP)
            cat, conf  = _apply_feedback(email, norm, conf, prefs)
            logger.debug(f"categorize:llm1 | {email_id} → {cat} {conf:.2f}")
            return {"category": cat, "confidence": conf}

    # ── Layer 2: LLM retry ────────────────────────────────────────────────────
    logger.warning(f"categorize:llm1 failed | {email_id}")
    prompt2 = (
        f'Email subject: "{email.get("subject", "")}"\n'
        f'Choose ONE from: {", ".join(_CATEGORIES_LIST)}\n'
        f'JSON only: {{"category": "NAME", "confidence": 0.70}}'
    )
    raw2    = call_llm(prompt2, temperature=0.0, max_tokens=60, use_cache=True)
    result2 = parse_json_strict(raw2, fallback={}, context=f"cat2_{email_id}")

    if validate_category_output(result2):
        norm = _normalize(result2.get("category", ""))
        if norm:
            conf       = min(float(result2.get("confidence", 0.65)), CONFIDENCE_CAP)
            cat, conf  = _apply_feedback(email, norm, conf, prefs)
            logger.info(f"categorize:llm2 | {email_id} → {cat}")
            return {"category": cat, "confidence": conf}

    # ── Layer 3: Heuristic fallback ───────────────────────────────────────────
    heur_cat, heur_conf = _heuristic_fallback(email)
    cat, conf           = _apply_feedback(email, heur_cat, heur_conf, prefs)
    logger.warning(f"categorize:heuristic | {email_id} → {cat}")
    return {"category": cat, "confidence": conf}
