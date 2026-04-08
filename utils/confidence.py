"""
Confidence calibration engine.
Replaces fake LLM self-reported confidence with
ensemble-based scoring using multiple signals.
"""
from config.constants import (
    CONFIDENCE_CAP, CONFIDENCE_BLEND_LLM, CONFIDENCE_BLEND_HEUR,
    URGENCY_KEYWORDS_HIGH, URGENCY_KEYWORDS_MEDIUM,
)
from config import SENDER_IMPORTANCE

# Keyword density per category
_CATEGORY_KEYWORDS = {
    "Alert / Urgent":        ["urgent","down","critical","asap","immediately","outage","503","not responding","alert","emergency","crashed"],
    "Action Required":       ["please","action","required","approval","approve","review","confirm","sign","submit","need","deadline"],
    "Newsletter":            ["unsubscribe","newsletter","digest","weekly","monthly","edition","issue #","read more","view online"],
    "Billing / Invoice":     ["invoice","payment","due","billing","subscription","renewal","charges","amount","account","renew"],
    "Meeting / Event":       ["meeting","calendar","invite","schedule","webinar","zoom","join","lunch","event","conference","rsvp"],
    "Social / Notification": ["linkedin","notification","connection","order","shipped","tracking","your account","new request"],
    "Travel":                ["flight","booking","pnr","hotel","trip","departure","arrives","check-in","itinerary","reservation"],
    "Job / Recruitment":     ["interview","position","role","candidate","hiring","resume","job","vacancy","apply","offer"],
    "General Info":          [],
}


def compute_heuristic_confidence(email: dict, category: str) -> float:
    """
    Multi-signal heuristic confidence scoring.
    Signals:
    1. Keyword density match for predicted category
    2. Sender importance
    3. Keyword exclusivity (keywords appear in ONLY this category)
    4. Subject line strength
    """
    text   = (email.get("subject","") + " " + email.get("body","")).lower()
    sender = email.get("sender","")
    subj   = email.get("subject","").lower()

    # Signal 1: keyword density
    kws        = _CATEGORY_KEYWORDS.get(category, [])
    matches    = sum(1 for kw in kws if kw in text)
    total_kws  = max(len(kws), 1)
    density    = min(matches / total_kws, 1.0)
    kw_score   = 0.50 + density * 0.30   # 0.50 to 0.80

    # Signal 2: sender importance boost
    sender_imp = SENDER_IMPORTANCE.get(sender, 0)
    if sender_imp >= 9:   sender_boost = 0.08
    elif sender_imp >= 7: sender_boost = 0.04
    else:                 sender_boost = 0.0

    # Signal 3: keyword exclusivity
    # Count how many other categories share these keywords
    other_matches = 0
    for other_cat, other_kws in _CATEGORY_KEYWORDS.items():
        if other_cat == category:
            continue
        other_matches += sum(1 for kw in other_kws if kw in text and kw in kws)
    exclusivity = 1.0 - min(other_matches / max(matches, 1), 1.0)
    excl_score  = exclusivity * 0.08

    # Signal 4: subject line contains category keywords
    subj_matches = sum(1 for kw in kws if kw in subj)
    subj_score   = min(subj_matches * 0.03, 0.06)

    raw_score = kw_score + sender_boost + excl_score + subj_score
    return round(min(raw_score, CONFIDENCE_CAP), 3)


def calibrate_confidence(
    llm_conf:      float,
    heuristic_conf: float,
    category:      str,
    email:         dict,
) -> float:
    """
    Ensemble confidence calibration.

    Method:
    1. Blend LLM and heuristic (weighted average)
    2. Apply plausibility penalty if LLM and heuristic disagree strongly
    3. Cap at CONFIDENCE_CAP to avoid overconfidence

    This produces more meaningful, spread-out confidence values
    rather than the repetitive 0.8/0.9 values from raw LLM output.
    """
    # Step 1: weighted blend
    blended = CONFIDENCE_BLEND_LLM * llm_conf + CONFIDENCE_BLEND_HEUR * heuristic_conf

    # Step 2: plausibility penalty
    # If LLM and heuristic disagree by more than 0.25 → penalize
    disagreement = abs(llm_conf - heuristic_conf)
    if disagreement > 0.25:
        penalty = disagreement * 0.15
        blended = blended - penalty

    # Step 3: domain-specific adjustments
    text = (email.get("subject","") + " " + email.get("body","")).lower()

    # Strong urgency signals → boost Alert/Urgent confidence
    if category == "Alert / Urgent":
        if any(kw in text for kw in ["urgent","down","critical","503","crashed"]):
            blended = min(blended + 0.10, CONFIDENCE_CAP)

    # Clear newsletter signals → boost Newsletter confidence
    if category == "Newsletter":
        if "unsubscribe" in text:
            blended = min(blended + 0.12, CONFIDENCE_CAP)

    # Known sender domain patterns
    sender = email.get("sender","").lower()
    domain_matches = {
        "billing":     "Billing / Invoice",
        "invoice":     "Billing / Invoice",
        "newsletter":  "Newsletter",
        "alert":       "Alert / Urgent",
        "monitor":     "Alert / Urgent",
        "noreply@linkedin": "Social / Notification",
        "shipping@":   "Social / Notification",
        "recruit":     "Job / Recruitment",
        "hr@":         "Job / Recruitment",
    }
    for pattern, expected_cat in domain_matches.items():
        if pattern in sender and expected_cat == category:
            blended = min(blended + 0.08, CONFIDENCE_CAP)
            break

    return round(min(max(blended, 0.30), CONFIDENCE_CAP), 3)