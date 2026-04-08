"""
Task extraction with HTML cleaning.
"""
import re
from datetime import datetime, date, timedelta
from utils.llm_client import call_llm
from utils.validators import parse_json_strict, validate_task_output
from utils.email_cleaner import clean_email_body
from utils.logger import setup_logger
from agent.prompts import DEFAULT_PROMPTS
from memory.repository import get_email, get_prompt, get_processed

logger = setup_logger(__name__)

VALID_TYPES = {"task","multi_step","reminder","calendar_event","informational"}

_DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\.?\s+(\d{1,2}),?\s+(\d{4})\b",
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\.?\s+(\d{1,2})\b",
]

_ALWAYS_EXTRACT = {
    "Action Required","Alert / Urgent","Billing / Invoice",
    "Job / Recruitment","Meeting / Event",
}
_NO_TASK = {"Newsletter","Social / Notification"}

_PHRASE_SIGNALS = [
    r"please\s+\w+", r"action\s+required", r"kindly\s+\w+",
    r"deadline\s+(is|was)", r"due\s+(date|by|on)",
    r"by\s+(friday|monday|eod|today|tomorrow)",
    r"feedback\s+(needed|required)", r"approve\s+(the|this)",
    r"deploy\s+(the|by)", r"must\s+be\s+(done|completed)",
]
_WORD_SIGNALS = [
    "urgent","asap","critical","deadline","approve","review",
    "confirm","sign","submit","attend","deploy","feedback","follow up",
]


def _signal_strength(text: str) -> str:
    t      = text.lower()
    phrases = sum(1 for p in _PHRASE_SIGNALS if re.search(p, t))
    words   = sum(1 for w in _WORD_SIGNALS if w in t)
    if phrases >= 2 or words >= 3: return "strong"
    if phrases >= 1 or words >= 1: return "weak"
    return "none"


def _parse_natural_deadline(text: str) -> str | None:
    today = date.today()
    t     = text.lower()

    m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if m:
        try:
            return str(datetime.strptime(m.group(1),"%Y-%m-%d").date())
        except ValueError:
            pass

    if re.search(r'\b(today|tonight|eod|end of day)\b', t):
        return str(today)
    if re.search(r'\btomorrow\b', t):
        return str(today + timedelta(days=1))

    days_map = {
        "monday":0,"tuesday":1,"wednesday":2,"thursday":3,
        "friday":4,"saturday":5,"sunday":6,
    }
    dm = re.search(
        r'\b(?:by\s+|this\s+|next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', t
    )
    if dm:
        target = days_map[dm.group(1)]
        ahead  = (target - today.weekday()) % 7 or 7
        return str(today + timedelta(days=ahead))

    m = re.search(r'\bin\s+(\d+)\s+days?\b', t)
    if m:
        return str(today + timedelta(days=int(m.group(1))))
    if re.search(r'\bnext\s+week\b', t):
        return str(today + timedelta(days=7))
    return None


def _validate_deadline(dl: str | None) -> str | None:
    if not dl:
        return None
    try:
        d = datetime.strptime(dl, "%Y-%m-%d").date()
        if (date.today() - d).days > 30:
            return None
        return dl
    except Exception:
        return None


_SUBJECT_PATTERNS = [
    (r'\bserver\b.*\b(down|not responding|outage)\b',
     "multi_step","Investigate and resolve server outage immediately",
     ["Check server logs","Restart affected service","Verify system health","Notify team"]),
    (r'\b(security\s+patch|cve|vulnerability|deploy)\b',
     "multi_step","Deploy the security patch to all environments",
     ["Review patch notes","Test in staging","Deploy to production","Confirm completion"]),
    (r'\b(approve|approval)\b',
     "task","Review and provide approval",None),
    (r'\b(sign|signature|docusign)\b',
     "task","Review and sign the required document",None),
    (r'\b(meeting|call|sync)\b',
     "calendar_event","Attend the scheduled meeting",None),
    (r'\b(interview)\b',
     "multi_step","Prepare for and attend the interview",
     ["Research the company","Prepare for questions","Confirm availability","Attend interview"]),
    (r'\b(invoice|bill)\b.*\b(due|payment)\b',
     "task","Process the invoice payment",None),
    (r'\bpull\s+request\b|\bpr\s*#\d+\b',
     "task","Review and merge the pull request",None),
    (r'\b(job|hiring|role|position)\b',
     "task","Review job opportunity and decide whether to apply",None),
    (r'\bfeedback\b.*\b(needed|required)\b',
     "task","Provide the requested feedback",None),
    (r'\b(flight|booking\s+confirmed)\b',
     "reminder","Keep travel documents ready for trip",None),
    (r'\b(performance\s+review)\b',
     "multi_step","Prepare for performance review",
     ["Complete self-assessment","Review achievements","Attend review meeting"]),
]

_CATEGORY_DEFAULTS = {
    "Action Required":       ("task",          "Review and complete the required action"),
    "Meeting / Event":       ("calendar_event","Attend the scheduled meeting or event"),
    "Billing / Invoice":     ("task",          "Review and process the invoice or payment"),
    "Alert / Urgent":        ("multi_step",    "Investigate and resolve the urgent alert"),
    "Job / Recruitment":     ("task",          "Review job opportunity and decide whether to apply"),
    "Travel":                ("reminder",      "Review travel details and keep ticket ready"),
    "General Info":          ("informational", "Review for relevant information"),
    "Newsletter":            ("informational", "Read later if relevant"),
    "Social / Notification": ("informational", "No action required"),
}


def _infer_task(subject: str, body: str, category: str = None) -> dict:
    text = (subject + " " + body[:200]).lower()
    for pattern, t_type, task, steps in _SUBJECT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {"type": t_type, "task": task, "steps": steps}
    if category in _CATEGORY_DEFAULTS:
        t_type, task = _CATEGORY_DEFAULTS[category]
        return {"type": t_type, "task": task, "steps": None}
    return {"type":"task", "task": f"Review: {subject[:60]}", "steps": None}


def _is_vague(task: str) -> bool:
    if not task:
        return True
    vague = ["review email","respond to","read email","check email"]
    tl    = task.lower().strip()
    return any(tl.startswith(v) for v in vague) or len(tl.split()) < 4


def _validate_output(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("type") not in VALID_TYPES:
        return False
    task = data.get("task","")
    return isinstance(task, str) and len(task.strip()) >= 5


def extract_tasks(email_id: str) -> dict:
    """
    Enriched task extraction with cleaned email body.
    """
    email = get_email(email_id)
    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    subject   = email.get("subject","")
    body      = email.get("body","")
    processed = get_processed(email_id) or {}
    category  = processed.get("category")

    # Clean body before any processing
    clean_body = clean_email_body(body, max_chars=500)
    full_clean = subject + " " + clean_body

    # Skip no-task categories
    if category in _NO_TASK:
        return {
            "task":      "No action required — informational only",
            "task_type": "informational",
            "steps":     None,
            "deadline":  None,
            "confidence": 0.95,
        }

    strength = _signal_strength(full_clean)
    use_llm  = (strength in ("strong","weak")) or (category in _ALWAYS_EXTRACT)

    llm_result = None
    if use_llm:
        from agent.prompts import DEFAULT_PROMPTS
        template = get_prompt("extract_tasks") or DEFAULT_PROMPTS["extract_tasks"]
        prompt   = template.format(subject=subject, body=clean_body[:400])
        raw      = call_llm(prompt, temperature=0.0, max_tokens=250, use_cache=True)
        parsed   = parse_json_strict(raw, fallback={}, context=f"tasks_{email_id}")
        if _validate_output(parsed):
            llm_result = parsed

    rule_result = _infer_task(subject, clean_body, category)

    if llm_result and not _is_vague(llm_result.get("task","")) \
            and len(llm_result.get("task","").split()) >= 5:
        best       = llm_result
        confidence = float(llm_result.get("confidence",0.82))
        source     = "llm"
    else:
        best       = rule_result
        confidence = 0.72 if strength != "none" else 0.58
        source     = "rule"

    # Boost if still vague
    if _is_vague(best.get("task","")):
        boost_prompt = (
            f'Email subject: "{subject}"\n'
            f'Body summary: "{clean_body[:200]}"\n'
            f'Write ONE specific actionable task (max 12 words).\n'
            f'Reply ONLY JSON: {{"task":"TASK","deadline":null,"confidence":0.7,"type":"task"}}'
        )
        raw2    = call_llm(boost_prompt, temperature=0.0, max_tokens=100)
        parsed2 = parse_json_strict(raw2, fallback={}, context=f"boost_{email_id}")
        if _validate_output(parsed2) and not _is_vague(parsed2.get("task","")):
            best       = parsed2
            confidence = float(parsed2.get("confidence",0.70))
            source     = "llm_boost"

    # Deadline
    rule_deadline = _parse_natural_deadline(subject) or _parse_natural_deadline(clean_body)
    llm_deadline  = best.get("deadline")
    if rule_deadline and llm_deadline:
        try:
            rd = datetime.strptime(rule_deadline, "%Y-%m-%d").date()
            ld = datetime.strptime(llm_deadline,  "%Y-%m-%d").date()
            raw_dl = str(min(rd, ld))
        except Exception:
            raw_dl = rule_deadline
    else:
        raw_dl = rule_deadline or llm_deadline

    final_deadline = _validate_deadline(raw_dl)
    task           = best.get("task","").strip()
    task_type      = best.get("type","task")
    steps          = best.get("steps")

    if task_type not in VALID_TYPES:
        task_type = "task"
    if isinstance(steps, list) and len(steps) == 0:
        steps = None

    logger.debug(
        f"extract_tasks | {email_id} source={source} "
        f"type={task_type} task={task!r} deadline={final_deadline}"
    )

    return {
        "task":       task,
        "task_type":  task_type,
        "steps":      steps,
        "deadline":   final_deadline,
        "confidence": round(confidence, 2),
    }