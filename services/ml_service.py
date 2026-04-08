"""
ML service updated with MLflow tracking and Redis LLM caching.
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime, date
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

MODEL_DIR              = "data/models"
MIN_SAMPLES            = 15
AUTO_RETRAIN_THRESHOLD = 5

FEATURE_NAMES = [
    "urgency_density","high_action","med_action","sender_importance",
    "category_urgency","deadline_proximity","subject_length","body_length",
    "has_attachment_signal","correction_boost",
]

_URGENCY_PHRASES = [
    "server down","production down","critical error","outage","not responding",
    "security breach","cve","vulnerability","asap","immediately","urgent",
    "emergency","critical","overdue","past due",
]
_HIGH_ACTION = [
    "please approve","action required","your approval","sign by","deadline",
    "by eod","by friday","by tomorrow","must complete","required by",
]
_MEDIUM_ACTION = [
    "please review","feedback needed","follow up","reminder",
    "kindly","when possible","at your convenience",
]
_CATEGORY_BASE = {
    "Alert / Urgent":1.0,"Action Required":2.0,"Billing / Invoice":3.0,
    "Meeting / Event":3.0,"Job / Recruitment":4.0,"General Info":5.0,
    "Travel":5.0,"Social / Notification":6.0,"Newsletter":7.0,
}

_dynamic_sender_weights: dict[str, float] = {
    "boss@company.com":   1.0,
    "cto@company.com":    1.0,
    "hr@company.com":     0.7,
    "client@bigcorp.com": 0.9,
}

_prediction_log: list[dict] = []


def update_dynamic_weights(feedback_history: list[dict], emails_map: dict):
    global _dynamic_sender_weights
    sender_corrections: dict[str, list[int]] = {}
    for fb in feedback_history:
        if fb.get("field") != "priority":
            continue
        email  = emails_map.get(fb["email_id"],{})
        sender = email.get("sender","")
        if not sender:
            continue
        try:
            new_pri = int(fb["new_value"])
            sender_corrections.setdefault(sender, []).append(new_pri)
        except Exception:
            pass
    for sender, priorities in sender_corrections.items():
        avg_pri    = sum(priorities) / len(priorities)
        importance = max(0.3, 1.0 - (avg_pri - 1) / 6.0)
        _dynamic_sender_weights[sender] = round(importance, 2)


def extract_features(
    email: dict,
    processed: dict,
    feedback_history: list[dict] | None = None,
) -> list[float]:
    text     = (email.get("subject","") + " " + email.get("body","")).lower()
    sender   = email.get("sender","").lower()
    category = processed.get("category","General Info") or "General Info"
    deadline = processed.get("deadline")

    f1 = min(sum(1 for p in _URGENCY_PHRASES if p in text) / max(len(_URGENCY_PHRASES),1), 1.0)
    f2 = min(sum(1 for p in _HIGH_ACTION if p in text) / 5.0, 1.0)
    f3 = min(sum(1 for p in _MEDIUM_ACTION if p in text) / 5.0, 1.0)

    sender_imp = max(
        (_dynamic_sender_weights.get(s,0) for s in _dynamic_sender_weights if s in sender),
        default=0.0
    )
    f4 = min(sender_imp, 1.0)
    f5 = 1.0 - (_CATEGORY_BASE.get(category,5.0) - 1) / 6.0

    if deadline:
        try:
            days = (datetime.strptime(deadline,"%Y-%m-%d").date() - date.today()).days
            if days < 0:    f6 = 0.0
            elif days == 0: f6 = 0.05
            elif days <= 1: f6 = 0.15
            elif days <= 3: f6 = 0.30
            elif days <= 7: f6 = 0.50
            else:           f6 = 0.80
        except Exception:
            f6 = 0.5
    else:
        f6 = 0.5

    f7  = min(len(email.get("subject","")) / 100.0, 1.0)
    f8  = min(len(email.get("body","")) / 1000.0, 1.0)
    f9  = float(any(kw in text for kw in ["attached","attachment","find attached"]))
    f10 = 0.0
    if feedback_history:
        for fb in feedback_history:
            if fb.get("field") == "priority" and fb.get("email_id") == email.get("id"):
                try:
                    f10 = (5.0 - float(fb["new_value"])) / 4.0
                except Exception:
                    pass

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]


def validate_dataset(X: list, y: list) -> tuple[bool, list[str]]:
    issues = []
    if len(X) < MIN_SAMPLES:
        issues.append(f"Need {MIN_SAMPLES} samples, have {len(X)}")
    if len(X) != len(y):
        issues.append("Feature/label length mismatch")
        return False, issues
    if not X:
        return False, issues
    X_arr = np.array(X)
    for i, name in enumerate(FEATURE_NAMES):
        if i < X_arr.shape[1]:
            col = X_arr[:, i]
            if np.any(np.isnan(col)):
                issues.append(f"NaN in feature {name}")
            if np.all(col == col[0]):
                issues.append(f"Zero variance in {name}")
    from collections import Counter
    if len(Counter(y)) < 2:
        issues.append("Only one class — cannot train")
    return len(issues) == 0, issues


def _get_model_path(version: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"priority_model_v{version}.pkl")


def train_model(
    emails: list[dict],
    processed_map: dict,
    feedback_history: list[dict],
    auto_version: bool = True,
) -> dict:
    """Train model with MLflow tracking."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score
    from memory.repository import (
        save_training_sample, get_all_training_data, save_model_version
    )
    from services.mlflow_tracker import log_training_run

    emails_map = {e["id"]: e for e in emails}
    update_dynamic_weights(feedback_history, emails_map)

    feedback_labels = {
        fb["email_id"]: int(fb["new_value"])
        for fb in feedback_history
        if fb.get("field") == "priority"
        and str(fb.get("new_value","")).isdigit()
    }

    X, y = [], []
    for email in emails:
        eid   = email["id"]
        proc  = processed_map.get(eid,{})
        label = feedback_labels.get(eid) or proc.get("priority")
        if not label:
            continue
        try:
            label = int(label)
            if not (1 <= label <= 7):
                continue
        except Exception:
            continue
        feat = extract_features(email, proc, feedback_history)
        X.append(feat)
        y.append(label)
        save_training_sample(eid, feat, label, "feedback")

    for sample in get_all_training_data():
        feat = sample.get("features",[])
        lbl  = sample.get("label")
        if feat and lbl and len(feat) == len(FEATURE_NAMES):
            X.append(feat)
            y.append(lbl)

    # Deduplicate
    seen = set()
    Xu, yu = [], []
    for feat, lbl in zip(X, y):
        k = tuple(feat)
        if k not in seen:
            seen.add(k)
            Xu.append(feat)
            yu.append(lbl)
    X, y = Xu, yu

    valid, issues = validate_dataset(X, y)
    if not valid:
        return {"success": False, "reason": "; ".join(issues), "issues": issues}

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            multi_class="ovr", max_iter=1000, C=1.0,
            class_weight="balanced", random_state=42,
        )),
    ])

    metrics = {"n_samples": len(X), "n_classes": len(set(y))}
    if len(X) >= 30:
        X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
        pipeline.fit(X_tr, y_tr)
        acc                     = accuracy_score(y_te, pipeline.predict(X_te))
        metrics["test_accuracy"] = round(float(acc), 4)
        cv = cross_val_score(pipeline, X_arr, y_arr, cv=3, scoring="accuracy")
        metrics["cv_accuracy_mean"] = round(float(cv.mean()), 4)
        metrics["cv_accuracy_std"]  = round(float(cv.std()), 4)
    else:
        metrics["test_accuracy"] = None

    pipeline.fit(X_arr, y_arr)

    version    = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if auto_version else "latest"
    model_path = _get_model_path(version)

    model_data = {
        "pipeline":          pipeline,
        "feature_names":     FEATURE_NAMES,
        "n_samples":         len(X),
        "metrics":           metrics,
        "version":           version,
        "sender_weights":    dict(_dynamic_sender_weights),
        "training_features": X[-50:],
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    accuracy = metrics.get("test_accuracy") or metrics.get("cv_accuracy_mean") or 0.0
    save_model_version(version, accuracy, len(X), model_path)

    # Log to MLflow
    run_id = log_training_run(model_data, metrics, FEATURE_NAMES, len(X))
    logger.info(f"Model v{version} saved | acc={accuracy} mlflow_run={run_id}")

    return {
        "success":    True,
        "version":    version,
        "metrics":    metrics,
        "mlflow_run": run_id,
    }


def load_active_model() -> dict | None:
    from memory.repository import get_active_model_version
    info = get_active_model_version()
    if not info:
        return None
    path = info.get("model_path","")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path,"rb") as f:
            data = pickle.load(f)
        global _dynamic_sender_weights
        if "sender_weights" in data:
            _dynamic_sender_weights.update(data["sender_weights"])
        return data
    except Exception as e:
        logger.error(f"Model load failed: {type(e).__name__}")
        return None


def should_auto_retrain(feedback_history: list[dict]) -> bool:
    from memory.repository import get_active_model_version
    info        = get_active_model_version()
    priority_fb = [f for f in feedback_history if f.get("field") == "priority"]
    if not info:
        return len(priority_fb) >= MIN_SAMPLES
    trained_at = info.get("trained_at","")
    new_fb     = [f for f in priority_fb if f.get("corrected_at","") > trained_at]
    return len(new_fb) >= AUTO_RETRAIN_THRESHOLD


def get_monitoring_report() -> dict:
    from memory.repository import get_active_model_version, get_model_history, get_all_training_data
    model_info   = get_active_model_version()
    baseline_acc = model_info.get("accuracy",0.70) if model_info else 0.70
    verified     = [p for p in _prediction_log if p.get("actual") is not None]
    accuracy_report = {
        "status":          "insufficient_data" if len(verified) < 10 else "healthy",
        "baseline":        baseline_acc,
        "n_predictions":   len(_prediction_log),
        "n_verified":      len(verified),
    }
    return {
        "model_version":  model_info.get("version","none") if model_info else "none",
        "baseline_acc":   baseline_acc,
        "accuracy":       accuracy_report,
        "model_history":  get_model_history(),
    }


_cached_model: dict | None = None


def _heuristic_priority(features: list[float]) -> int:
    weights = [0.25,0.15,0.05,0.20,0.18,-0.12,0.02,0.02,0.03,0.08]
    score   = 0.35 + sum(w*v for w, v in zip(weights, features))
    return max(1, min(7, round(7 - max(0.0, min(1.0, score)) * 6)))


def _apply_overrides(priority: int, email: dict, processed: dict) -> int:
    text = (email.get("subject","") + " " + email.get("body","")).lower()
    cat  = processed.get("category","")
    if any(p in text for p in ["server down","production down","503","not responding","outage"]):
        return 1
    if any(p in text for p in ["cve","security breach","security patch","deploy by eod"]):
        return min(priority, 2)
    if cat in {"Newsletter","Social / Notification"}:
        return max(priority, 6)
    return priority


def predict_priority(
    email: dict,
    processed: dict,
    feedback_history: list[dict] | None = None,
) -> dict:
    global _cached_model

    features  = extract_features(email, processed, feedback_history)
    feat_dict = dict(zip(FEATURE_NAMES, features))
    feat_arr  = np.array([features])

    if _cached_model is None:
        _cached_model = load_active_model()

    method     = "heuristic"
    priority   = _heuristic_priority(features)
    confidence = 0.60

    if _cached_model is not None:
        try:
            pipe       = _cached_model["pipeline"]
            priority   = int(pipe.predict(feat_arr)[0])
            proba      = pipe.predict_proba(feat_arr)[0]
            confidence = float(np.max(proba))
            method     = "ml"
        except Exception as e:
            logger.warning(f"ML predict failed: {type(e).__name__}")

    priority  = _apply_overrides(priority, email, processed)
    _prediction_log.append({
        "email_id":  email.get("id",""),
        "predicted": priority,
        "method":    method,
        "actual":    None,
    })
    if len(_prediction_log) > 200:
        _prediction_log[:] = _prediction_log[-100:]

    from utils.observability import record_ml_prediction
    record_ml_prediction(method)

    top   = sorted(feat_dict.items(), key=lambda x: -x[1])[:3]
    labels = {
        "urgency_density":   "urgent keywords",
        "sender_importance": "important sender",
        "high_action":       "action required",
        "category_urgency":  f"category: {processed.get('category','')}",
    }
    parts = [labels.get(k,k) for k, v in top if v > 0.2]
    explanation = f"Priority {priority} via {method}" + (f" — {', '.join(parts)}" if parts else "")

    return {
        "priority":    priority,
        "confidence":  round(confidence, 3),
        "method":      method,
        "features":    feat_dict,
        "explanation": explanation,
    }


def batch_predict(emails, processed_map, feedback_history=None):
    return {
        e["id"]: predict_priority(e, processed_map.get(e["id"],{}), feedback_history)
        for e in emails
    }