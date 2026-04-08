"""
Background processing with DB-backed job persistence.
Jobs survive server restarts.
Uses ThreadPoolExecutor — no Celery/Redis needed.
"""
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.secure_logger import get_secure_logger
from utils.helpers import utcnow_iso
from memory.repository import create_job, update_job, get_job

logger = get_secure_logger(__name__)

MAX_WORKERS = 4


def process_emails_background(
    emails: list[dict],
    persona: str  = "Formal",
    dry_run: bool = False,
    job_id: str   | None = None,
) -> str:
    """
    Process emails in background using ThreadPoolExecutor.
    Job state persisted to SQLite — survives restarts.
    Returns job_id for status polling.
    """
    from services.email_service import process_email

    if job_id is None:
        job_id = f"proc_{uuid.uuid4().hex[:8]}"

    create_job(job_id, "process_emails", total=len(emails))
    logger.info(f"Job {job_id} created | emails={len(emails)}")

    def run():
        completed = 0
        errors    = 0
        results   = []

        update_job(job_id, status="running")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_email, email["id"], persona, dry_run): email["id"]
                for email in emails
            }
            for future in as_completed(futures):
                email_id = futures[future]
                try:
                    result = future.result(timeout=90)
                    if "error" in result:
                        errors += 1
                    else:
                        completed += 1
                    results.append({"id": email_id, "ok": "error" not in result})
                except Exception as e:
                    errors += 1
                    logger.error(f"Thread error {email_id}: {type(e).__name__}")
                    results.append({"id": email_id, "ok": False})

                total    = len(emails)
                progress = round((completed + errors) / total * 100)
                update_job(
                    job_id,
                    completed=completed,
                    errors=errors,
                    progress=progress,
                )

        update_job(
            job_id,
            status="done",
            completed=completed,
            errors=errors,
            progress=100,
            result_json=json.dumps(results),
            finished_at=utcnow_iso(),
        )
        logger.info(f"Job {job_id} done | ok={completed} errors={errors}")

    thread = threading.Thread(target=run, daemon=True, name=f"job_{job_id}")
    thread.start()
    return job_id


def fetch_gmail_background(
    access_token: str,
    job_id: str     | None = None,
    max_results: int = 20,
) -> str:
    """Fetch Gmail in background. DB-persisted job."""
    if job_id is None:
        job_id = f"gmail_{uuid.uuid4().hex[:8]}"

    create_job(job_id, "gmail_fetch", total=max_results)

    def run():
        update_job(job_id, status="running")
        try:
            from services.email_service import import_gmail_emails
            emails = import_gmail_emails(access_token, max_results)
            update_job(
                job_id,
                status="done",
                completed=len(emails),
                progress=100,
                result_json=json.dumps({"count": len(emails)}),
                finished_at=utcnow_iso(),
            )
            logger.info(f"Gmail job {job_id} done | count={len(emails)}")
        except Exception as e:
            update_job(
                job_id,
                status="error",
                error_msg=str(type(e).__name__),
                progress=100,
                finished_at=utcnow_iso(),
            )
            logger.error(f"Gmail job {job_id} failed: {type(e).__name__}")

    thread = threading.Thread(target=run, daemon=True, name=f"job_{job_id}")
    thread.start()
    return job_id


def poll_job(job_id: str, timeout: int = 120) -> dict:
    """
    Poll job status until done or timeout.
    Returns final job dict.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        if not job:
            return {"status": "not_found"}
        if job.get("status") in ("done","error"):
            return job
        time.sleep(1.0)
    return get_job(job_id) or {"status": "timeout"}