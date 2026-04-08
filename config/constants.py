CATEGORIES = [
    "Action Required",
    "Meeting / Event",
    "Newsletter",
    "Billing / Invoice",
    "Alert / Urgent",
    "Social / Notification",
    "Travel",
    "Job / Recruitment",
    "General Info",
]

PRIORITY_LEVELS = {
    1: "Critical", 2: "High", 3: "Medium", 4: "Low",
    5: "Very Low",  6: "Minimal", 7: "Negligible",
}

PRIORITY_LABELS = {
    1: "🔴 Critical", 2: "🟠 High",    3: "🟡 Medium",
    4: "🔵 Low",      5: "⚪ Very Low", 6: "⚪ Minimal", 7: "⚪ Negligible",
}

CATEGORY_PRIORITY_MAP = {
    "Alert / Urgent":        1,
    "Action Required":       2,
    "Billing / Invoice":     3,
    "Meeting / Event":       3,
    "Job / Recruitment":     4,
    "General Info":          5,
    "Travel":                5,
    "Social / Notification": 6,
    "Newsletter":            7,
}

SKIP_REPLY_CATEGORIES  = {"Newsletter", "Social / Notification", "Travel", "Alert / Urgent"}
SKIP_TASK_CATEGORIES   = {"Newsletter", "Social / Notification", "Travel"}
NO_PROCESS_CATEGORIES  = {"Newsletter", "Social / Notification"}

# ── Urgency keywords with weights ─────────────────────────────────────────────
URGENCY_KEYWORDS_HIGH = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "down", "outage", "not responding", "503", "production",
    "breach", "attack", "failure", "crashed",
]

URGENCY_KEYWORDS_MEDIUM = [
    "deadline", "action required", "end of day", "eod",
    "today", "right away", "please review", "approval needed",
    "by friday", "by monday", "by tomorrow",
]
URGENCY_KEYWORDS = [
    *URGENCY_KEYWORDS_HIGH,
    *URGENCY_KEYWORDS_MEDIUM
]

# ── Confidence calibration ─────────────────────────────────────────────────────
CONFIDENCE_HIGH          = 0.85
CONFIDENCE_LOW           = 0.50
CONFIDENCE_CAP           = 0.88
CONFIDENCE_BLEND_LLM     = 0.55
CONFIDENCE_BLEND_HEUR    = 0.45

# ── Confidence bands for display ──────────────────────────────────────────────
CONFIDENCE_BAND_HIGH     = 0.80   # ✅ auto-handle
CONFIDENCE_BAND_MED      = 0.60   # ⚠️ review suggested
# below 0.60 → ❌ needs review

# ── Cache settings ────────────────────────────────────────────────────────────
ENABLE_CACHE             = True
CACHE_TTL_SECONDS        = 3600   # 1 hour

# ── Snooze intelligence ───────────────────────────────────────────────────────
SNOOZE_DEFAULT_HOURS     = 24
SNOOZE_LOW_PRIORITY_HOURS = 48
SNOOZE_NEWSLETTER_HOURS  = 72

# ── Performance ───────────────────────────────────────────────────────────────
MAX_RETRIES              = 1
GROQ_MODEL               = "llama-3.1-8b-instant"
LOG_FILE                 = "logs/aeoa.log"
LOG_LEVEL                = "DEBUG"

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS                 = ["Formal", "Friendly", "Concise"]
DEFAULT_PERSONA          = "Formal"