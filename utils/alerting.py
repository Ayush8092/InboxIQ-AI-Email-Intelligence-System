"""
Alerting system with configurable rules.
- High error rate alerts
- Latency spike alerts
- ML accuracy degradation alerts
- Feature drift alerts
- Rate limit breach alerts
Notifications via: logs, Prometheus labels, optional webhook.
"""
import os
import time
import json
import requests
from dataclasses import dataclass, field
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")  # optional Slack/Teams webhook


@dataclass
class AlertRule:
    name:        str
    description: str
    threshold:   float
    window_s:    int      = 300   # evaluation window in seconds
    severity:    str      = "warning"   # info | warning | critical
    cooldown_s:  int      = 600   # minimum seconds between repeat alerts


@dataclass
class Alert:
    rule:       AlertRule
    value:      float
    message:    str
    fired_at:   float = field(default_factory=time.time)
    resolved:   bool  = False


# ── Alert rule definitions ────────────────────────────────────────────────────

ALERT_RULES = {
    "high_error_rate": AlertRule(
        name="high_error_rate",
        description="API error rate exceeds threshold",
        threshold=0.10,    # 10% error rate
        window_s=300,
        severity="critical",
        cooldown_s=300,
    ),
    "latency_spike": AlertRule(
        name="latency_spike",
        description="API p95 latency exceeds threshold",
        threshold=5.0,     # 5 seconds
        window_s=60,
        severity="warning",
        cooldown_s=120,
    ),
    "ml_accuracy_drop": AlertRule(
        name="ml_accuracy_drop",
        description="ML model accuracy dropped significantly",
        threshold=0.15,    # 15% drop
        window_s=3600,
        severity="critical",
        cooldown_s=1800,
    ),
    "feature_drift_high": AlertRule(
        name="feature_drift_high",
        description="High PSI feature drift detected",
        threshold=0.25,
        window_s=3600,
        severity="warning",
        cooldown_s=3600,
    ),
    "rate_limit_breach": AlertRule(
        name="rate_limit_breach",
        description="Excessive rate limit violations",
        threshold=10,      # 10 violations in window
        window_s=300,
        severity="warning",
        cooldown_s=600,
    ),
    "feedback_quality_low": AlertRule(
        name="feedback_quality_low",
        description="User feedback rejection rate is high",
        threshold=0.50,    # 50% feedback rejected
        window_s=3600,
        severity="info",
        cooldown_s=3600,
    ),
}

# State tracking
_active_alerts:   dict[str, Alert] = {}
_last_fired:      dict[str, float] = {}
_metric_windows:  dict[str, list]  = {}   # {rule_name: [(ts, value), ...]}


def _record_metric(rule_name: str, value: float):
    """Record a metric value with timestamp."""
    now = time.time()
    if rule_name not in _metric_windows:
        _metric_windows[rule_name] = []
    _metric_windows[rule_name].append((now, value))

    # Prune old values outside window
    rule   = ALERT_RULES.get(rule_name)
    window = rule.window_s if rule else 300
    cutoff = now - window
    _metric_windows[rule_name] = [
        (ts, v) for ts, v in _metric_windows[rule_name] if ts >= cutoff
    ]


def _should_fire(rule: AlertRule) -> bool:
    """Check if alert is in cooldown."""
    last = _last_fired.get(rule.name, 0)
    return (time.time() - last) >= rule.cooldown_s


def _fire_alert(rule: AlertRule, value: float, context: str = ""):
    """Fire an alert — log, store, and optionally notify."""
    if not _should_fire(rule):
        return

    message = (
        f"[{rule.severity.upper()}] {rule.name}: {rule.description}. "
        f"Value={value:.4f} Threshold={rule.threshold:.4f}. {context}"
    )

    alert = Alert(rule=rule, value=value, message=message)
    _active_alerts[rule.name] = alert
    _last_fired[rule.name]    = time.time()

    # Log at appropriate level
    if rule.severity == "critical":
        logger.error(f"ALERT FIRED: {message}")
    elif rule.severity == "warning":
        logger.warning(f"ALERT FIRED: {message}")
    else:
        logger.info(f"ALERT FIRED: {message}")

    # Optional webhook notification
    if ALERT_WEBHOOK_URL:
        _send_webhook(rule, value, message)


def _resolve_alert(rule_name: str):
    """Resolve an active alert."""
    if rule_name in _active_alerts:
        _active_alerts[rule_name].resolved = True
        logger.info(f"Alert resolved: {rule_name}")
        del _active_alerts[rule_name]


def _send_webhook(rule: AlertRule, value: float, message: str):
    """Send alert to Slack/Teams webhook (optional)."""
    try:
        emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(rule.severity, "⚪")
        payload = {
            "text": f"{emoji} *AEOA Alert* — {message}",
            "attachments": [{
                "color":  "danger" if rule.severity == "critical" else "warning",
                "fields": [
                    {"title": "Rule",      "value": rule.name,       "short": True},
                    {"title": "Severity",  "value": rule.severity,   "short": True},
                    {"title": "Value",     "value": str(round(value,4)), "short": True},
                    {"title": "Threshold", "value": str(rule.threshold), "short": True},
                ],
            }],
        }
        requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.warning(f"Webhook send failed: {type(e).__name__}")


# ── Alert check functions ─────────────────────────────────────────────────────

def check_error_rate(total_requests: int, error_requests: int):
    """Check if API error rate exceeds threshold."""
    if total_requests == 0:
        return
    rate = error_requests / total_requests
    _record_metric("high_error_rate", rate)
    rule = ALERT_RULES["high_error_rate"]
    if rate >= rule.threshold:
        _fire_alert(rule, rate, f"Errors={error_requests} Total={total_requests}")
    else:
        _resolve_alert("high_error_rate")


def check_latency(p95_latency_s: float):
    """Check if p95 API latency exceeds threshold."""
    rule = ALERT_RULES["latency_spike"]
    if p95_latency_s >= rule.threshold:
        _fire_alert(rule, p95_latency_s, f"p95={p95_latency_s:.2f}s")
    else:
        _resolve_alert("latency_spike")


def check_ml_accuracy(current_accuracy: float, baseline_accuracy: float):
    """Check if ML accuracy has dropped significantly."""
    drop = baseline_accuracy - current_accuracy
    rule = ALERT_RULES["ml_accuracy_drop"]
    if drop >= rule.threshold:
        _fire_alert(
            rule, drop,
            f"Baseline={baseline_accuracy:.3f} Current={current_accuracy:.3f}"
        )
    else:
        _resolve_alert("ml_accuracy_drop")


def check_feature_drift(max_psi: float):
    """Check if any feature has high PSI drift."""
    rule = ALERT_RULES["feature_drift_high"]
    if max_psi >= rule.threshold:
        _fire_alert(rule, max_psi, f"Max PSI={max_psi:.4f}")
    else:
        _resolve_alert("feature_drift_high")


def check_feedback_quality(
    total_feedback: int, rejected_feedback: int
):
    """Check if feedback rejection rate is too high."""
    if total_feedback == 0:
        return
    rate = rejected_feedback / total_feedback
    rule = ALERT_RULES["feedback_quality_low"]
    if rate >= rule.threshold:
        _fire_alert(
            rule, rate,
            f"Rejected={rejected_feedback}/{total_feedback}"
        )
    else:
        _resolve_alert("feedback_quality_low")


def get_active_alerts() -> list[dict]:
    """Return all currently active (unfired) alerts."""
    return [
        {
            "name":      a.rule.name,
            "severity":  a.rule.severity,
            "message":   a.message,
            "value":     a.value,
            "fired_at":  a.fired_at,
        }
        for a in _active_alerts.values()
        if not a.resolved
    ]


def get_alert_history() -> list[dict]:
    """Return recent alert metric history."""
    return {
        name: [{"ts": ts, "value": v} for ts, v in vals[-20:]]
        for name, vals in _metric_windows.items()
    }