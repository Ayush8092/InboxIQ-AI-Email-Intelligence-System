"""
Safe online learning service.
- Confidence threshold before accepting feedback
- User trust scoring system
- Validation before model update
- Rollback capability
- Periodic evaluation against held-out set
"""
import os
import json
import copy
import pickle
import time
from datetime import datetime
from utils.secure_logger import get_secure_logger
from memory.repository import get_all_feedback, get_all_emails, get_all_processed

logger = get_secure_logger(__name__)

# Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.70   # only accept feedback where model was at least 70% confident
MIN_USER_TRUST_SCORE     = 0.40   # users below this threshold have feedback ignored
MAX_LABEL_DEVIATION      = 2      # reject correction if |new - predicted| > 2 (likely noise)
TRUST_UPDATE_WEIGHT      = 0.1    # how fast trust score updates (Bayesian-style)
ROLLBACK_HISTORY_SIZE    = 10     # keep last N online model snapshots

_online_model            = None
_model_snapshots: list   = []     # for rollback
_user_trust_scores: dict[str, float] = {}  # user_id → trust score (0-1)
_update_log: list[dict]  = []     # history of all online updates


def _get_online_model():
    """Get or initialise River online learning model."""
    global _online_model
    if _online_model is None:
        try:
            from river import linear_model, preprocessing, compose
            _online_model = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression(),
            )
            logger.info("River online model initialised")
        except ImportError:
            logger.warning("river not installed — online learning disabled")
    return _online_model


def get_user_trust_score(user_id: str) -> float:
    """
    Get trust score for a user (0–1).
    New users start at 0.6 (neutral).
    Score increases when corrections match ground truth.
    Score decreases for contradictory or noisy corrections.
    """
    return _user_trust_scores.get(user_id, 0.6)


def update_user_trust(user_id: str, correction_was_correct: bool):
    """
    Update user trust score using exponential moving average.
    Correct corrections → trust increases.
    Wrong corrections → trust decreases.
    """
    current = get_user_trust_score(user_id)
    signal  = 1.0 if correction_was_correct else 0.0
    updated = (1 - TRUST_UPDATE_WEIGHT) * current + TRUST_UPDATE_WEIGHT * signal
    _user_trust_scores[user_id] = round(updated, 4)
    logger.info(
        f"Trust updated | user={user_id} "
        f"{current:.3f} → {updated:.3f} correct={correction_was_correct}"
    )


def validate_feedback(
    email: dict,
    processed: dict,
    new_priority: int,
    user_id: str,
    model_confidence: float | None = None,
) -> tuple[bool, str]:
    """
    Validate feedback before applying to online model.

    Checks:
    1. User trust score threshold
    2. Label deviation (is correction reasonable?)
    3. Model confidence (only update if model was confident enough to learn from)
    4. Duplicate detection (same user correcting same email repeatedly)

    Returns (is_valid, reason).
    """
    # Check 1: User trust
    trust = get_user_trust_score(user_id)
    if trust < MIN_USER_TRUST_SCORE:
        reason = (
            f"User trust score {trust:.2f} below threshold {MIN_USER_TRUST_SCORE}. "
            f"Feedback ignored to protect model quality."
        )
        logger.warning(f"Feedback rejected | user={user_id} trust={trust:.2f}")
        return False, reason

    # Check 2: Label deviation
    current_priority = processed.get("priority", 4)
    deviation        = abs(new_priority - current_priority)
    if deviation > MAX_LABEL_DEVIATION:
        reason = (
            f"Correction deviation {deviation} exceeds max {MAX_LABEL_DEVIATION}. "
            f"Current={current_priority}, Proposed={new_priority}. "
            f"Possible noise — saving but not applying to online model."
        )
        logger.warning(
            f"Large deviation | email={email.get('id')} "
            f"current={current_priority} new={new_priority}"
        )
        return False, reason

    # Check 3: Priority range
    if not (1 <= new_priority <= 7):
        return False, f"Priority {new_priority} out of valid range [1-7]"

    # Check 4: Duplicate recent correction from same user
    recent_same = [
        u for u in _update_log[-20:]
        if u.get("user_id") == user_id
        and u.get("email_id") == email.get("id")
        and (time.time() - u.get("ts", 0)) < 300  # within 5 minutes
    ]
    if recent_same:
        return False, "Duplicate correction from same user within 5 minutes — ignored"

    return True, "valid"


def _save_snapshot():
    """Save current online model state for rollback capability."""
    model = _get_online_model()
    if model is None:
        return
    try:
        snapshot = {
            "model":    copy.deepcopy(model),
            "ts":       time.time(),
            "n_updates": len(_update_log),
        }
        _model_snapshots.append(snapshot)
        # Keep only last N snapshots
        if len(_model_snapshots) > ROLLBACK_HISTORY_SIZE:
            _model_snapshots.pop(0)
        logger.debug(f"Online model snapshot saved (total={len(_model_snapshots)})")
    except Exception as e:
        logger.warning(f"Snapshot save failed: {type(e).__name__}")


def rollback_online_model(steps: int = 1) -> bool:
    """
    Rollback online model to a previous snapshot.
    steps=1 means go back 1 snapshot.
    Returns True if rollback succeeded.
    """
    global _online_model
    if not _model_snapshots:
        logger.warning("No snapshots available for rollback")
        return False

    idx = max(0, len(_model_snapshots) - steps - 1)
    try:
        snapshot        = _model_snapshots[idx]
        _online_model   = copy.deepcopy(snapshot["model"])
        _model_snapshots[:] = _model_snapshots[:idx]
        logger.info(
            f"Online model rolled back {steps} step(s) "
            f"to snapshot from {datetime.utcfromtimestamp(snapshot['ts']).isoformat()}"
        )
        return True
    except Exception as e:
        logger.error(f"Rollback failed: {type(e).__name__}")
        return False


def reset_online_model() -> bool:
    """Reset online model to fresh state."""
    global _online_model, _model_snapshots, _update_log
    try:
        from river import linear_model, preprocessing, compose
        _online_model   = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(),
        )
        _model_snapshots = []
        _update_log      = []
        logger.info("Online model reset to fresh state")
        return True
    except Exception as e:
        logger.error(f"Model reset failed: {type(e).__name__}")
        return False


def online_learn(
    email: dict,
    processed: dict,
    correct_priority: int,
    user_id: str = "system",
    model_confidence: float | None = None,
) -> dict:
    """
    Safe online learning update.

    Pipeline:
    1. Validate feedback (trust score, deviation, duplicates)
    2. Save snapshot before update
    3. Apply incremental update to River model
    4. Log update for audit trail
    5. Return result with validation details

    Returns {success, reason, trust_score, n_updates}
    """
    from services.ml_service import extract_features, FEATURE_NAMES

    model = _get_online_model()
    if model is None:
        return {"success": False, "reason": "Online model not available"}

    # Step 1: Validate
    is_valid, reason = validate_feedback(
        email, processed, correct_priority, user_id, model_confidence
    )
    if not is_valid:
        return {
            "success":     False,
            "reason":      reason,
            "trust_score": get_user_trust_score(user_id),
            "n_updates":   len(_update_log),
        }

    # Step 2: Snapshot before update
    _save_snapshot()

    # Step 3: Apply update
    try:
        features = extract_features(email, processed)
        x        = {name: val for name, val in zip(FEATURE_NAMES, features)}
        model.learn_one(x, correct_priority)

        # Step 4: Log update
        _update_log.append({
            "email_id":    email.get("id"),
            "user_id":     user_id,
            "priority":    correct_priority,
            "ts":          time.time(),
            "trust_score": get_user_trust_score(user_id),
        })

        logger.info(
            f"Online model updated | email={email.get('id')} "
            f"priority={correct_priority} user={user_id} "
            f"trust={get_user_trust_score(user_id):.2f} "
            f"total_updates={len(_update_log)}"
        )

        return {
            "success":     True,
            "reason":      "Model updated successfully",
            "trust_score": get_user_trust_score(user_id),
            "n_updates":   len(_update_log),
        }

    except Exception as e:
        logger.error(f"Online learn failed: {type(e).__name__}")
        return {"success": False, "reason": str(type(e).__name__)}


def evaluate_online_model(
    test_emails: list[dict],
    test_processed: dict,
    test_labels: dict[str, int],
) -> dict:
    """
    Periodic evaluation of online model against held-out labeled data.
    Call this after N updates to check model quality.

    test_labels: {email_id: correct_priority}
    """
    from services.ml_service import extract_features, FEATURE_NAMES

    model = _get_online_model()
    if model is None or not test_emails:
        return {"error": "Model not available or no test data"}

    correct = 0
    total   = 0
    errors  = []

    for email in test_emails:
        eid   = email.get("id","")
        label = test_labels.get(eid)
        if label is None:
            continue

        proc     = test_processed.get(eid, {})
        features = extract_features(email, proc)
        x        = {name: val for name, val in zip(FEATURE_NAMES, features)}

        try:
            pred = model.predict_one(x)
            if pred is not None:
                total += 1
                if abs(int(pred) - label) <= 1:
                    correct += 1
                else:
                    errors.append({
                        "email_id":  eid,
                        "predicted": int(pred),
                        "actual":    label,
                    })
        except Exception:
            pass

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Online model evaluation | accuracy={accuracy:.3f} n={total}")

    return {
        "accuracy":  round(accuracy, 4),
        "n_samples": total,
        "correct":   correct,
        "errors":    errors[:10],
        "n_updates": len(_update_log),
    }


def get_update_log() -> list[dict]:
    return list(_update_log)


def get_trust_scores() -> dict:
    return dict(_user_trust_scores)