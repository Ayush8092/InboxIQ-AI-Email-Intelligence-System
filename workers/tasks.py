"""
All Celery tasks.
Each task is:
  - Idempotent (Redis key check)
  - Retryable with exponential backoff
  - Progress-reporting via Redis pub/sub
  - Observability-instrumented
"""
import time
import json
from celery import shared_task
from workers.celery_app import celery_app
from utils.secure_logger import get_secure_logger
from utils.redis_client import (
    mark_email_processed, is_email_processed,
    set_job_status, publish_job_progress,
)

logger = get_secure_logger(__name__)


# ── Email processing tasks ────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="workers.tasks.process_email_task",
    max_retries=3,
    default_retry_delay=5,
    queue="email_processing",
)
def process_email_task(self, email_id: str, persona: str = "Formal", dry_run: bool = False):
    """
    Process a single email through the full pipeline.
    Idempotent — skips if already processed.
    Retries up to 3 times with 5s backoff.
    """
    from services.email_service import process_email
    from utils.observability import record_email_processed

    # Idempotency check
    if is_email_processed(email_id):
        logger.info(f"Idempotency hit — skipping: {email_id}")
        return {"status": "skipped", "reason": "already_processed", "email_id": email_id}

    try:
        result = process_email(email_id, persona=persona, dry_run=dry_run)

        if "error" not in result:
            # Mark as processed in Redis
            summary = {
                "category": result.get("final_state",{}).get("category"),
                "priority":  result.get("final_state",{}).get("priority"),
                "ts":        time.time(),
            }
            mark_email_processed(email_id, summary)
            record_email_processed("success")
        else:
            record_email_processed("error")

        return result

    except Exception as e:
        logger.error(f"process_email_task failed | email_id={email_id} error={e}")
        record_email_processed("error")
        raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))


@celery_app.task(
    bind=True,
    name="workers.tasks.process_all_task",
    max_retries=1,
    queue="email_processing",
)
def process_all_task(self, persona: str = "Formal", dry_run: bool = False):
    """
    Process all emails using a Celery group (fan-out).
    Dispatches one process_email_task per email.
    Reports progress via Redis pub/sub.
    """
    from memory.repository import get_all_emails
    from celery import group

    job_id = self.request.id
    emails = get_all_emails()
    total  = len(emails)

    set_job_status(job_id, {
        "status":    "running",
        "total":     total,
        "completed": 0,
        "errors":    0,
        "progress":  0,
    })

    logger.info(f"process_all_task | job_id={job_id} total={total}")

    # Create task group — all emails processed in parallel
    tasks = group(
        process_email_task.s(
            email["id"], persona, dry_run
        )
        for email in emails
        if not is_email_processed(email["id"])
    )

    result = tasks.apply_async()
    completed = 0
    errors    = 0

    for r in result.iterate(disable_sync_subtasks=False):
        if isinstance(r, dict) and "error" not in r:
            completed += 1
        else:
            errors += 1

        progress = round((completed + errors) / total * 100) if total > 0 else 100
        publish_job_progress(job_id, progress, f"Processed {completed}/{total}")
        set_job_status(job_id, {
            "status":    "running",
            "total":     total,
            "completed": completed,
            "errors":    errors,
            "progress":  progress,
        })

    set_job_status(job_id, {
        "status":    "done",
        "total":     total,
        "completed": completed,
        "errors":    errors,
        "progress":  100,
    })

    logger.info(f"process_all_task done | ok={completed} errors={errors}")
    return {"completed": completed, "errors": errors, "total": total}


@celery_app.task(
    bind=True,
    name="workers.tasks.fetch_gmail_task",
    max_retries=3,
    queue="email_processing",
)
def fetch_gmail_task(self, encrypted_token: str, max_results: int = 50):
    """
    Fetch Gmail emails in background.
    access_token is passed encrypted — decrypted only here.
    """
    from utils.encryption import decrypt_token
    from utils.gmail_fetcher import fetch_gmail_emails
    from memory.repository import insert_email

    job_id = self.request.id
    set_job_status(job_id, {"status": "fetching", "progress": 0})

    try:
        access_token = decrypt_token(encrypted_token)
        if not access_token:
            raise ValueError("Token decryption failed")

        emails = fetch_gmail_emails(access_token, max_results=max_results)
        inserted = 0
        for email in emails:
            try:
                insert_email(email)
                inserted += 1
            except Exception as e:
                logger.warning(f"Insert failed for {email.get('id')}: {type(e).__name__}")

        set_job_status(job_id, {
            "status":   "done",
            "progress": 100,
            "count":    inserted,
        })
        logger.info(f"fetch_gmail_task done | inserted={inserted}")
        return {"status": "done", "count": inserted}

    except Exception as e:
        set_job_status(job_id, {"status": "error", "error": str(type(e).__name__)})
        raise self.retry(exc=e, countdown=10 * (self.request.retries + 1))


# ── ML tasks ──────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="workers.tasks.train_model_task",
    max_retries=1,
    queue="ml_tasks",
    soft_time_limit=240,
)
def train_model_task(self, simulate_feedback: bool = False, n_simulate: int = 20):
    """Train ML model in background."""
    from memory.repository import get_all_emails, get_all_processed, get_all_feedback
    from services.ml_service import train_model
    from services.feedback_simulator import simulate_feedback as _sim

    job_id = self.request.id
    set_job_status(job_id, {"status": "training", "progress": 10})

    try:
        if simulate_feedback:
            samples = _sim(n_samples=n_simulate)
            logger.info(f"Simulated {len(samples)} feedback samples")
            set_job_status(job_id, {"status": "training", "progress": 30})

        emails   = get_all_emails()
        proc_map = {p["email_id"]: p for p in get_all_processed()}
        feedback = get_all_feedback()
        result   = train_model(emails, proc_map, feedback)

        set_job_status(job_id, {
            "status":   "done",
            "progress": 100,
            "result":   result,
        })

        if result.get("success"):
            from utils.observability import set_ml_accuracy
            acc = (result.get("metrics",{}).get("test_accuracy") or
                   result.get("metrics",{}).get("cv_accuracy_mean") or 0.0)
            set_ml_accuracy(acc)

        return result

    except Exception as e:
        set_job_status(job_id, {"status": "error", "error": str(type(e).__name__)})
        raise self.retry(exc=e, countdown=30)


@celery_app.task(
    bind=True,
    name="workers.tasks.online_learn_task",
    max_retries=2,
    queue="ml_tasks",
)
def online_learn_task(
    self,
    email_id: str,
    correct_priority: int,
    user_id: str = "system",
):
    """Apply online learning update in background."""
    from memory.repository import get_email, get_processed
    from services.online_learning import online_learn

    try:
        email  = get_email(email_id)
        proc   = get_processed(email_id) or {}
        if not email:
            return {"success": False, "reason": "email not found"}

        result = online_learn(email, proc, correct_priority, user_id=user_id)
        logger.info(
            f"online_learn_task | email={email_id} "
            f"priority={correct_priority} result={result.get('success')}"
        )
        return result

    except Exception as e:
        raise self.retry(exc=e, countdown=5)


# ── Periodic/beat tasks ────────────────────────────────────────────────────────

@celery_app.task(name="workers.tasks.check_drift_task", queue="ml_tasks")
def check_drift_task():
    """Hourly drift check — runs via Celery Beat."""
    from memory.repository import get_all_training_data, get_all_processed
    from services.ml_service import _prediction_log, FEATURE_NAMES
    from services.drift_detector import unified_drift_report
    from utils.alerting import check_feature_drift, check_ml_accuracy
    from memory.repository import get_active_model_version

    train_data   = get_all_training_data()
    processed    = get_all_processed()
    ref_features = [d["features"] for d in train_data[-100:] if d.get("features")]
    ref_labels   = [d["label"] for d in train_data[-100:] if d.get("label")]
    cur_labels   = [p.get("priority",4) for p in processed[-50:] if p.get("priority")]
    cur_features = ref_features[-20:] if len(ref_features) > 20 else ref_features

    if not ref_features:
        return {"status": "skipped", "reason": "no training data"}

    report = unified_drift_report(
        reference_features=ref_features,
        current_features=cur_features,
        feature_names=FEATURE_NAMES,
        predictions=_prediction_log,
        reference_labels=ref_labels,
        current_labels=cur_labels,
    )

    if report.get("overall_severity") == "high":
        max_psi = max(
            (v.get("psi",0) for v in report.get("feature_drift",{}).get("features",{}).values()),
            default=0
        )
        check_feature_drift(max_psi)

    logger.info(f"Drift check complete | severity={report.get('overall_severity')}")
    return {"status": "done", "severity": report.get("overall_severity")}


@celery_app.task(name="workers.tasks.evaluate_model_task", queue="ml_tasks")
def evaluate_model_task():
    """Periodic model evaluation — runs via Celery Beat every 6 hours."""
    from memory.repository import get_all_emails, get_all_processed, get_all_feedback
    from services.online_learning import evaluate_online_model

    emails    = get_all_emails()
    proc_map  = {p["email_id"]: p for p in get_all_processed()}
    feedback  = get_all_feedback()

    # Build ground truth from feedback
    labels = {
        fb["email_id"]: int(fb["new_value"])
        for fb in feedback
        if fb.get("field") == "priority"
        and str(fb.get("new_value","")).isdigit()
    }

    if not labels:
        return {"status": "skipped", "reason": "no labeled data"}

    result = evaluate_online_model(emails, proc_map, labels)
    logger.info(f"Model evaluation | accuracy={result.get('accuracy')} n={result.get('n_samples')}")

    from utils.alerting import check_ml_accuracy
    from memory.repository import get_active_model_version
    model_info   = get_active_model_version()
    baseline_acc = model_info.get("accuracy", 0.70) if model_info else 0.70
    check_ml_accuracy(result.get("accuracy", 0.0), baseline_acc)

    return result