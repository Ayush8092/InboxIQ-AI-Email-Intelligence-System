"""
Observability layer.
- Prometheus metrics (counters, histograms, gauges)
- Structured JSON logging for log aggregation
- Request tracing
"""
import time
import json
import logging
import os
from functools import wraps
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
_prometheus_available = False
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    )
    _prometheus_available = True
except ImportError:
    logger.warning("prometheus_client not available — metrics disabled")

if _prometheus_available:
    _registry = CollectorRegistry()

    EMAILS_PROCESSED = Counter(
        "aeoa_emails_processed_total",
        "Total emails processed",
        ["status"],
        registry=_registry,
    )
    LLM_REQUESTS = Counter(
        "aeoa_llm_requests_total",
        "Total LLM API calls",
        ["tool", "status"],
        registry=_registry,
    )
    LLM_LATENCY = Histogram(
        "aeoa_llm_latency_seconds",
        "LLM call latency",
        ["tool"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        registry=_registry,
    )
    API_REQUESTS = Counter(
        "aeoa_api_requests_total",
        "Total API requests",
        ["endpoint", "method", "status"],
        registry=_registry,
    )
    API_LATENCY = Histogram(
        "aeoa_api_latency_seconds",
        "API endpoint latency",
        ["endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        registry=_registry,
    )
    ML_PREDICTIONS = Counter(
        "aeoa_ml_predictions_total",
        "Total ML predictions",
        ["method"],
        registry=_registry,
    )
    ML_ACCURACY = Gauge(
        "aeoa_ml_accuracy",
        "Current ML model accuracy",
        registry=_registry,
    )
    ACTIVE_JOBS = Gauge(
        "aeoa_active_jobs",
        "Currently running background jobs",
        registry=_registry,
    )
    CACHE_HITS = Counter(
        "aeoa_cache_hits_total",
        "Cache hit count",
        ["cache_type"],
        registry=_registry,
    )
    FEEDBACK_CORRECTIONS = Counter(
        "aeoa_feedback_corrections_total",
        "User feedback corrections",
        ["field"],
        registry=_registry,
    )
    RATE_LIMIT_HITS = Counter(
        "aeoa_rate_limit_hits_total",
        "Rate limit exceeded count",
        ["action"],
        registry=_registry,
    )


def get_metrics() -> bytes:
    """Return Prometheus metrics in text format."""
    if not _prometheus_available:
        return b"# Prometheus not available\n"
    return generate_latest(_registry)


def get_content_type() -> str:
    if not _prometheus_available:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# ── Metric recording helpers ──────────────────────────────────────────────────

def record_email_processed(status: str = "success"):
    if _prometheus_available:
        EMAILS_PROCESSED.labels(status=status).inc()


def record_llm_call(tool: str, latency_s: float, status: str = "success"):
    if _prometheus_available:
        LLM_REQUESTS.labels(tool=tool, status=status).inc()
        LLM_LATENCY.labels(tool=tool).observe(latency_s)


def record_api_request(endpoint: str, method: str, status: int, latency_s: float):
    if _prometheus_available:
        API_REQUESTS.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        API_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def record_ml_prediction(method: str):
    if _prometheus_available:
        ML_PREDICTIONS.labels(method=method).inc()


def set_ml_accuracy(accuracy: float):
    if _prometheus_available:
        ML_ACCURACY.set(accuracy)


def record_cache_hit(cache_type: str = "llm"):
    if _prometheus_available:
        CACHE_HITS.labels(cache_type=cache_type).inc()


def record_feedback(field: str):
    if _prometheus_available:
        FEEDBACK_CORRECTIONS.labels(field=field).inc()


def record_rate_limit(action: str):
    if _prometheus_available:
        RATE_LIMIT_HITS.labels(action=action).inc()


# ── Structured logging ────────────────────────────────────────────────────────

class StructuredLogger:
    """
    JSON structured logger for log aggregation.
    Each log entry is a JSON object with consistent fields.
    """

    def __init__(self, name: str):
        self._logger = get_secure_logger(name)
        self._name   = name

    def _log(self, level: str, event: str, **fields):
        record = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level":   level,
            "logger":  self._name,
            "event":   event,
            **{k: v for k, v in fields.items() if v is not None},
        }
        msg = json.dumps(record, default=str)
        getattr(self._logger, level.lower(), self._logger.info)(msg)

    def info(self, event: str, **fields):
        self._log("INFO", event, **fields)

    def warning(self, event: str, **fields):
        self._log("WARNING", event, **fields)

    def error(self, event: str, **fields):
        self._log("ERROR", event, **fields)

    def debug(self, event: str, **fields):
        self._log("DEBUG", event, **fields)


# ── Tracing decorator ─────────────────────────────────────────────────────────

def trace(operation_name: str):
    """
    Decorator for tracing function execution.
    Logs entry, exit, latency, and any exceptions.
    """
    def decorator(fn):
        _slog = StructuredLogger(fn.__module__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start     = time.time()
            trace_id  = os.urandom(4).hex()
            _slog.debug(
                f"{operation_name}.start",
                trace_id=trace_id,
                fn=fn.__name__,
            )
            try:
                result  = fn(*args, **kwargs)
                latency = time.time() - start
                _slog.info(
                    f"{operation_name}.complete",
                    trace_id=trace_id,
                    latency_ms=round(latency * 1000),
                )
                return result
            except Exception as e:
                latency = time.time() - start
                _slog.error(
                    f"{operation_name}.error",
                    trace_id=trace_id,
                    latency_ms=round(latency * 1000),
                    error=type(e).__name__,
                )
                raise

        return wrapper
    return decorator


def get_system_health() -> dict:
    """
    Return overall system health summary.
    Used by /health endpoint and dashboard.
    """
    from memory.repository import (
        get_all_emails, get_all_processed, get_total_llm_calls,
        get_all_feedback, get_recent_jobs, get_active_model_version,
    )
    from utils.cache import cache_stats

    emails    = get_all_emails()
    processed = get_all_processed()
    feedback  = get_all_feedback()
    jobs      = get_recent_jobs(5)
    model     = get_active_model_version()
    cs        = cache_stats()

    failed_jobs = [j for j in jobs if j.get("status") == "error"]

    return {
        "status": "healthy" if not failed_jobs else "degraded",
        "emails": {
            "total":      len(emails),
            "processed":  len(processed),
            "needs_review": sum(1 for p in processed if p.get("needs_review")),
        },
        "ml": {
            "model_active":  model is not None,
            "model_version": model.get("version","none") if model else "none",
            "model_accuracy": model.get("accuracy",0) if model else 0,
        },
        "cache": cs,
        "feedback": {
            "total":    len(feedback),
            "priority": len([f for f in feedback if f.get("field")=="priority"]),
        },
        "recent_jobs": [
            {"id": j["id"], "type": j["job_type"], "status": j["status"]}
            for j in jobs
        ],
    }