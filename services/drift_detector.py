"""
Unified drift detection combining:
1. Feature drift (PSI — Population Stability Index)
2. Concept drift (accuracy degradation over sliding window)
3. Label distribution drift (output label shift)
4. Unified alerting with severity levels
"""
import numpy as np
from collections import Counter
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

# Thresholds
PSI_HIGH       = 0.25   # significant feature drift
PSI_MODERATE   = 0.10   # moderate feature drift
ACCURACY_DROP  = 0.15   # accuracy drop triggers alert
LABEL_DRIFT_KL = 0.20   # KL-divergence threshold for label distribution
WINDOW_SIZE    = 50     # sliding window for concept drift detection


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI = Σ (current_% - ref_%) * ln(current_% / ref_%)
    PSI < 0.1: no drift
    PSI 0.1–0.25: moderate drift
    PSI > 0.25: significant drift
    """
    eps          = 1e-8
    bin_edges    = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()) + eps,
        bins + 1,
    )
    ref_hist, _  = np.histogram(reference, bins=bin_edges)
    cur_hist, _  = np.histogram(current,   bins=bin_edges)
    ref_pct      = (ref_hist + eps) / len(reference)
    cur_pct      = (cur_hist + eps) / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _kl_divergence(p: dict, q: dict) -> float:
    """
    KL Divergence between two label distributions.
    p = reference distribution, q = current distribution.
    """
    all_labels = set(p) | set(q)
    eps        = 1e-8
    total_p    = sum(p.values()) or 1
    total_q    = sum(q.values()) or 1
    kl         = 0.0
    for label in all_labels:
        p_prob = p.get(label, 0) / total_p + eps
        q_prob = q.get(label, 0) / total_q + eps
        kl    += p_prob * np.log(p_prob / q_prob)
    return float(kl)


def detect_feature_drift(
    reference_features: list[list[float]],
    current_features:   list[list[float]],
    feature_names:      list[str],
) -> dict:
    """
    Feature drift detection using PSI per feature.
    reference_features: training data feature matrix
    current_features:   recent inference feature matrix
    """
    if not reference_features or not current_features:
        return {"drift_detected": False, "features": {}, "error": "insufficient data"}

    ref = np.array(reference_features)
    cur = np.array(current_features)

    if ref.shape[1] != cur.shape[1]:
        return {"drift_detected": False, "error": "feature dimension mismatch"}

    results     = {}
    high_drift  = []
    mod_drift   = []

    for i, name in enumerate(feature_names):
        psi      = _psi(ref[:, i], cur[:, i])
        severity = "none"
        if psi >= PSI_HIGH:
            severity = "high"
            high_drift.append(name)
        elif psi >= PSI_MODERATE:
            severity = "moderate"
            mod_drift.append(name)

        results[name] = {
            "psi":      round(psi, 4),
            "severity": severity,
            "ref_mean": round(float(np.mean(ref[:, i])), 4),
            "cur_mean": round(float(np.mean(cur[:, i])), 4),
            "ref_std":  round(float(np.std(ref[:, i])), 4),
            "cur_std":  round(float(np.std(cur[:, i])), 4),
        }

    drift_detected = len(high_drift) > 0

    if drift_detected:
        logger.warning(
            f"Feature drift detected | high={high_drift} moderate={mod_drift}"
        )

    return {
        "drift_detected":        drift_detected,
        "features":              results,
        "n_high_drift":          len(high_drift),
        "n_moderate_drift":      len(mod_drift),
        "high_drift_features":   high_drift,
        "moderate_drift_features": mod_drift,
    }


def detect_concept_drift(
    predictions: list[dict],
    window_size: int = WINDOW_SIZE,
) -> dict:
    """
    Concept drift = model accuracy is degrading over time.
    Uses sliding window comparison:
    - First half accuracy vs second half accuracy

    predictions: list of {predicted, actual} where actual is known.
    """
    verified = [p for p in predictions if p.get("actual") is not None]
    if len(verified) < window_size // 2:
        return {
            "drift_detected": False,
            "status":         "insufficient_data",
            "n_verified":     len(verified),
        }

    recent   = verified[-window_size:]
    mid      = len(recent) // 2
    first    = recent[:mid]
    second   = recent[mid:]

    def acc(samples):
        if not samples:
            return 0.0
        return sum(
            1 for s in samples
            if abs(s.get("predicted",0) - s.get("actual",0)) <= 1
        ) / len(samples)

    first_acc  = acc(first)
    second_acc = acc(second)
    drop       = first_acc - second_acc
    drift      = drop > ACCURACY_DROP

    if drift:
        logger.warning(
            f"Concept drift detected | "
            f"first_half_acc={first_acc:.3f} "
            f"second_half_acc={second_acc:.3f} "
            f"drop={drop:.3f}"
        )

    return {
        "drift_detected":   drift,
        "first_half_acc":   round(first_acc, 4),
        "second_half_acc":  round(second_acc, 4),
        "accuracy_drop":    round(drop, 4),
        "window_size":      len(recent),
        "n_verified":       len(verified),
        "severity":         "high" if drop > ACCURACY_DROP * 1.5 else ("moderate" if drift else "none"),
    }


def detect_label_drift(
    reference_labels: list[int],
    current_labels:   list[int],
) -> dict:
    """
    Label distribution drift — has the output priority distribution shifted?
    E.g., suddenly 80% of emails are being predicted as Critical.
    Uses KL divergence between reference and current label distributions.
    """
    if not reference_labels or not current_labels:
        return {"drift_detected": False, "error": "insufficient data"}

    ref_dist = Counter(reference_labels)
    cur_dist = Counter(current_labels)
    kl       = _kl_divergence(ref_dist, cur_dist)
    drift    = kl > LABEL_DRIFT_KL

    if drift:
        logger.warning(
            f"Label distribution drift | KL={kl:.4f} "
            f"ref={dict(ref_dist)} cur={dict(cur_dist)}"
        )

    return {
        "drift_detected":      drift,
        "kl_divergence":       round(kl, 4),
        "reference_dist":      dict(ref_dist),
        "current_dist":        dict(cur_dist),
        "severity":            "high" if kl > LABEL_DRIFT_KL * 2 else ("moderate" if drift else "none"),
    }


def unified_drift_report(
    reference_features: list[list[float]],
    current_features:   list[list[float]],
    feature_names:      list[str],
    predictions:        list[dict],
    reference_labels:   list[int],
    current_labels:     list[int],
) -> dict:
    """
    Run all three drift detectors and combine into unified report.
    Returns overall severity and actionable recommendations.
    """
    feature_drift = detect_feature_drift(
        reference_features, current_features, feature_names
    )
    concept_drift = detect_concept_drift(predictions)
    label_drift   = detect_label_drift(reference_labels, current_labels)

    any_drift = (
        feature_drift.get("drift_detected") or
        concept_drift.get("drift_detected") or
        label_drift.get("drift_detected")
    )

    # Determine overall severity
    severities = []
    for report in [feature_drift, concept_drift, label_drift]:
        s = report.get("severity","none")
        if s == "high":
            severities.append(2)
        elif s == "moderate":
            severities.append(1)
        else:
            severities.append(0)

    max_severity = max(severities) if severities else 0
    overall      = ["none","moderate","high"][max_severity]

    # Recommendations
    recommendations = []
    if feature_drift.get("drift_detected"):
        recommendations.append(
            f"Feature drift detected in "
            f"{feature_drift.get('n_high_drift',0)} features — "
            f"retrain with recent data"
        )
    if concept_drift.get("drift_detected"):
        drop = concept_drift.get("accuracy_drop",0)
        recommendations.append(
            f"Concept drift: accuracy dropped {drop:.1%} — "
            f"immediate retraining recommended"
        )
    if label_drift.get("drift_detected"):
        recommendations.append(
            f"Label distribution shifted (KL={label_drift.get('kl_divergence',0):.3f}) — "
            f"check for data bias or system changes"
        )
    if not recommendations:
        recommendations.append("No drift detected — model is stable")

    return {
        "any_drift":       any_drift,
        "overall_severity": overall,
        "feature_drift":   feature_drift,
        "concept_drift":   concept_drift,
        "label_drift":     label_drift,
        "recommendations": recommendations,
    }