"""
Celery application configuration.
Broker:  Redis (queue management)
Backend: Redis (result storage)
Queues:
  - email_processing: Gmail fetch + email pipeline
  - ml_tasks:         model training, online learning
  - default:          general tasks
"""
import os
from celery import Celery
from celery.signals import task_failure, task_success, worker_ready
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_BACKEND_URL = os.getenv("REDIS_BACKEND_URL", "redis://localhost:6379/1")

celery_app = Celery(
    "aeoa",
    broker=REDIS_URL,
    backend=REDIS_BACKEND_URL,
    include=["workers.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer          = "json",
    accept_content           = ["json"],
    result_serializer        = "json",

    # Routing — dedicated queues per task type
    task_routes = {
        "workers.tasks.process_email_task":  {"queue": "email_processing"},
        "workers.tasks.process_all_task":    {"queue": "email_processing"},
        "workers.tasks.fetch_gmail_task":    {"queue": "email_processing"},
        "workers.tasks.train_model_task":    {"queue": "ml_tasks"},
        "workers.tasks.online_learn_task":   {"queue": "ml_tasks"},
    },

    # Reliability
    task_acks_late           = True,    # ACK only after task completes
    task_reject_on_worker_lost = True,  # requeue if worker dies
    worker_prefetch_multiplier = 1,     # one task at a time per worker slot
    task_max_retries         = 3,
    task_default_retry_delay = 5,       # seconds

    # Performance
    result_expires           = 86400,   # 24 hours
    task_soft_time_limit     = 300,     # 5 min soft limit
    task_time_limit          = 360,     # 6 min hard limit

    # Beat schedule (periodic tasks)
    beat_schedule = {
        "check-drift-every-hour": {
            "task":     "workers.tasks.check_drift_task",
            "schedule": 3600,   # every hour
            "options":  {"queue": "ml_tasks"},
        },
        "evaluate-model-every-6h": {
            "task":     "workers.tasks.evaluate_model_task",
            "schedule": 21600,  # every 6 hours
            "options":  {"queue": "ml_tasks"},
        },
    },

    timezone                 = "UTC",
    enable_utc               = True,
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logger.info(f"Celery worker ready | queues={sender.app.amqp.queues}")


@task_failure.connect
def on_task_failure(task_id, exception, **kwargs):
    logger.error(
        f"Task failed | task_id={task_id} "
        f"error={type(exception).__name__}: {exception}"
    )


@task_success.connect
def on_task_success(sender, result, **kwargs):
    logger.debug(f"Task succeeded | task={sender.name}")