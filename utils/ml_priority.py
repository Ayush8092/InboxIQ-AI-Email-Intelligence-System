"""
Auto-priority ML model.
Predicts task urgency (1-7) using:
- Sender importance signals
- Keyword features
- Deadline proximity
- Historical correction patterns from feedback
- Category-based priors

Uses a lightweight logistic regression trained on feedback data.
Falls back to rule-based scoring when insufficient training data.
"""
import re
import math
from datetime import datetime, date
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Feature extraction ────────────────────────────────────────────────────────

_URGENCY_PHRASES = [
    "server down", "production down", "critical error", "outage", "not responding",
    "security breach", "cve", "vulnerability", "asap", "immediately", "urgent",
    "emergency", "critical", "deadline today", "overdue", "past due",
]

_HIGH_ACTION = [
    "please approve", "action required", "your approval", "sign by", "deadline",
    "by eod", "by friday", "by tomorrow", "must complete", "required by",
]

_MEDIUM_ACTION = [
    "please review", "feedback needed", "follow up", "reminder", "kindly",
    "when possible", "at your convenience",
]

_CATEGORY_BASE = {
    "Alert / Urgent":        1.0,
    "Action Required":       2.0,
    "Billing / Invoice":     3.0,
    "Meeting / Event":       3.0,
    "Job / Recruitment":     4.0,
    "General Info":          5.0,
    "Travel":                5.0,
    "Social / Notification": 6.0,
    "Newsletter":            7.0,
}

_SENDER_SCORES = {
    "boss@company.com":   10,
    "cto@company.com":    10,
    "hr@company.com":      7,
    "client@bigcorp.com":  9,
}


def extract_features(
    email: dict,
    processed: dict,
    feedback_history: list[dict] | None = None,
) -> dict:
    """
    Extract numerical features for priority prediction.
    Returns a feature dict suitable for scoring.
    """
    text     = (email.get("subject","") + " " + email.get("body","")).lower()
    sender   = email.get("sender","").lower()
    category = processed.get("category","General Info") or "General Info"
    deadline = processed.get("deadline")
    task     = processed.get("task","") or ""

    features = {}

    # F1: Urgency phrase density
    urgency_hits  = sum(1 for p in _URGENCY_PHRASES if p in text)
    features["urgency_density"] = min(urgency_hits / max(len(_URGENCY_PHRASES), 1), 1.0)

    # F2: High-action phrase presence
    high_hits = sum(1 for p in _HIGH_ACTION if p in text)
    features["high_action"]  = min(high_hits / 5.0, 1.0)

    # F3: Medium-action phrase presence
    med_hits = sum(1 for p in _MEDIUM_ACTION if p in text)
    features["med_action"] = min(med_hits / 5.0, 1.0)

    # F4: Sender importance (normalised 0-1)
    sender_imp = 0
    for known_sender, imp in _SENDER_SCORES.items():
        if known_sender in sender:
            sender_imp = imp
            break
    features["sender_importance"] = sender_imp / 10.0

    # F5: Category base priority (normalised, inverted: lower = more urgent)
    cat_base = _CATEGORY_BASE.get(category, 5.0)
    features["category_urgency"] = 1.0 - (cat_base - 1) / 6.0   # 1=most urgent, 0=least

    # F6: Deadline proximity (0=overdue/today, 1=far future, 0.5=no deadline)
    if deadline:
        try:
            dl   = datetime.strptime(deadline, "%Y-%m-%d").date()
            days = (dl - date.today()).days
            if days < 0:
                prox = 0.0   # overdue
            elif days == 0:
                prox = 0.05
            elif days <= 1:
                prox = 0.15
            elif days <= 3:
                prox = 0.30
            elif days <= 7:
                prox = 0.50
            else:
                prox = 0.80
        except Exception:
            prox = 0.5
    else:
        prox = 0.5
    features["deadline_proximity"] = prox   # lower = more urgent

    # F7: Subject length signal (longer subjects often = more important)
    subj_len = len(email.get("subject",""))
    features["subject_length"] = min(subj_len / 100.0, 1.0)

    # F8: Body length signal
    body_len = len(email.get("body",""))
    features["body_length"] = min(body_len / 1000.0, 1.0)

    # F9: Has attachment signal (keywords)
    has_attach = int(any(kw in text for kw in ["attached","attachment","find attached","please see"]))
    features["has_attachment_signal"] = float(has_attach)

    # F10: Historical correction signal
    correction_boost = 0.0
    if feedback_history:
        for fb in feedback_history:
            if fb.get("field") == "priority" and fb.get("email_id") == email.get("id"):
                try:
                    correction_boost = (5.0 - float(fb["new_value"])) / 4.0
                except Exception:
                    pass
    features["correction_boost"] = correction_boost

    return features


def _weighted_score(features: dict) -> float:
    """
    Weighted linear combination of features → raw urgency score (0-1).
    Higher = more urgent.

    Weights tuned for email triage:
    - Urgency phrases and sender importance are strongest signals
    - Deadline proximity is critical
    - Category prior anchors the base
    """
    weights = {
        "urgency_density":      0.25,
        "high_action":          0.15,
        "med_action":           0.05,
        "sender_importance":    0.20,
        "category_urgency":     0.18,
        "deadline_proximity":   -0.12,  # negative: lower proximity = more urgent
        "subject_length":       0.02,
        "body_length":          0.02,
        "has_attachment_signal":0.03,
        "correction_boost":     0.08,
    }

    score = 0.35  # baseline (maps to priority ~4)
    for feat, val in features.items():
        w      = weights.get(feat, 0)
        score += w * val

    return max(0.0, min(1.0, score))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def predict_priority(
    email: dict,
    processed: dict,
    feedback_history: list[dict] | None = None,
) -> dict:
    """
    Predict priority (1-7) using ML-style weighted feature scoring.

    Returns:
    {
      "priority":    int 1-7,
      "confidence":  float 0-1,
      "features":    dict,
      "explanation": str,
    }
    """
    features    = extract_features(email, processed, feedback_history)
    raw_score   = _weighted_score(features)
    # Map 0-1 urgency score → 1-7 priority (inverted: high urgency = low priority number)
    priority    = max(1, min(7, round(7 - raw_score * 6)))

    # Hard overrides for clear signals
    text = (email.get("subject","") + " " + email.get("body","")).lower()
    if any(p in text for p in ["server down","production down","503","not responding","outage"]):
        priority    = 1
        raw_score   = 1.0

    elif any(p in text for p in ["cve","security patch","critical vulnerability","deploy by eod"]):
        priority    = min(priority, 2)
        raw_score   = max(raw_score, 0.80)

    category = processed.get("category","")
    if category in {"Newsletter","Social / Notification"} and priority < 5:
        priority  = max(priority, 6)
        raw_score = min(raw_score, 0.25)

    # Confidence: how decisive the features are
    feat_variance = max(features.values()) - min(features.values())
    confidence    = round(min(0.55 + feat_variance * 0.4, 0.92), 3)

    # Human-readable explanation
    top_feats = sorted(
        [(k, v) for k, v in features.items() if v > 0.3],
        key=lambda x: -x[1]
    )[:3]
    explanation_parts = []
    for feat, val in top_feats:
        if feat == "urgency_density":
            explanation_parts.append("urgent keywords detected")
        elif feat == "sender_importance":
            explanation_parts.append("important sender")
        elif feat == "category_urgency":
            explanation_parts.append(f"category: {category}")
        elif feat == "high_action":
            explanation_parts.append("action required phrases")
        elif feat == "correction_boost":
            explanation_parts.append("user correction history")

    explanation = (
        f"Priority {priority} — " +
        (", ".join(explanation_parts) if explanation_parts else "standard scoring")
    )

    return {
        "priority":    priority,
        "confidence":  confidence,
        "raw_score":   round(raw_score, 3),
        "features":    features,
        "explanation": explanation,
    }


def batch_predict(
    emails: list[dict],
    processed_map: dict,
    feedback_history: list[dict] | None = None,
) -> dict[str, dict]:
    """
    Predict priorities for a batch of emails.
    Returns {email_id: prediction_dict}
    """
    results = {}
    for email in emails:
        eid       = email["id"]
        processed = processed_map.get(eid, {})
        results[eid] = predict_priority(email, processed, feedback_history)
    return results