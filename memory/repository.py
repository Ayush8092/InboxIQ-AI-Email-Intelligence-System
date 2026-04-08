import json
from utils.helpers import utcnow_iso
from utils.secure_logger import get_secure_logger
from memory.db import get_connection

logger = get_secure_logger(__name__)


# ── Emails ────────────────────────────────────────────────────────────────────

def insert_email(email: dict, source: str = "demo"):
    """
    Insert email with source tag.
    source = 'demo'  → preloaded demo emails
    source = 'gmail' → real Gmail emails fetched via OAuth
    """
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO emails "
        "(id, subject, body, sender, timestamp, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            email["id"],
            email["subject"],
            email["body"],
            email["sender"],
            email["timestamp"],
            email.get("source", source),
        ),
    )
    conn.commit()
    conn.close()


def get_email(email_id: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM emails WHERE id=?", (email_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_emails(source: str | None = None) -> list[dict]:
    """
    Get emails filtered by source.
    source=None      → all emails
    source='demo'    → only demo emails
    source='gmail'   → only Gmail emails
    """
    conn = get_connection()
    if source:
        rows = conn.execute(
            "SELECT * FROM emails WHERE source=? ORDER BY timestamp DESC",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM emails ORDER BY timestamp DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_email_sources() -> list[str]:
    """Return all distinct sources present in DB."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT source FROM emails"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]


def clear_emails_by_source(source: str):
    """
    Delete all emails of a given source.
    Used to clear demo emails after Gmail login.
    """
    conn = get_connection()
    conn.execute("DELETE FROM emails WHERE source=?", (source,))
    conn.commit()
    conn.close()
    logger.info(f"Cleared all emails with source='{source}'")


# ── Processed ─────────────────────────────────────────────────────────────────

def upsert_processed(data: dict):
    steps     = data.get("steps")
    steps_str = json.dumps(steps) if isinstance(steps, list) else (steps or "")

    conn = get_connection()
    conn.execute("""
        INSERT INTO processed
            (email_id, category, priority, task, task_type, steps,
             deadline, summary, confidence, needs_review, review_reason)
        VALUES
            (:email_id, :category, :priority, :task, :task_type, :steps,
             :deadline, :summary, :confidence, :needs_review, :review_reason)
        ON CONFLICT(email_id) DO UPDATE SET
            category=excluded.category, priority=excluded.priority,
            task=excluded.task, task_type=excluded.task_type,
            steps=excluded.steps, deadline=excluded.deadline,
            summary=excluded.summary, confidence=excluded.confidence,
            needs_review=excluded.needs_review,
            review_reason=excluded.review_reason
    """, {
        "email_id":      data.get("email_id"),
        "category":      data.get("category"),
        "priority":      data.get("priority", 3),
        "task":          data.get("task"),
        "task_type":     data.get("task_type", "task"),
        "steps":         steps_str,
        "deadline":      data.get("deadline"),
        "summary":       data.get("summary"),
        "confidence":    data.get("confidence"),
        "needs_review":  data.get("needs_review", 0),
        "review_reason": data.get("review_reason", ""),
    })
    conn.commit()
    conn.close()


def get_processed(email_id: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM processed WHERE email_id=?", (email_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["steps"] = json.loads(d["steps"]) if d.get("steps") else []
    except Exception:
        d["steps"] = []
    return d


def get_all_processed(source: str | None = None) -> list[dict]:
    """
    Get processed results optionally filtered by email source.
    source='gmail' → only processed Gmail emails
    source='demo'  → only processed demo emails
    """
    conn = get_connection()
    if source:
        rows = conn.execute("""
            SELECT p.* FROM processed p
            JOIN emails e ON p.email_id = e.id
            WHERE e.source = ?
        """, (source,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM processed").fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["steps"] = json.loads(d["steps"]) if d.get("steps") else []
        except Exception:
            d["steps"] = []
        result.append(d)
    return result


def update_processed_field(email_id: str, field: str, value):
    allowed = {
        "category", "priority", "task", "task_type", "steps", "deadline",
        "summary", "confidence", "needs_review", "review_reason", "snoozed_until",
    }
    if field not in allowed:
        raise ValueError(f"Field '{field}' not updatable.")
    conn = get_connection()
    conn.execute(
        f"UPDATE processed SET {field}=? WHERE email_id=?", (value, email_id)
    )
    conn.commit()
    conn.close()


# ── Drafts ────────────────────────────────────────────────────────────────────

def insert_draft(email_id: str, subject: str, body: str, persona: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO drafts (email_id,subject,body,persona,created_at) VALUES (?,?,?,?,?)",
        (email_id, subject, body, persona, utcnow_iso()),
    )
    conn.commit()
    conn.close()


def get_drafts(email_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM drafts WHERE email_id=? ORDER BY created_at DESC",
        (email_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_drafts() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM drafts ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_prompt(name: str) -> str | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT template FROM prompts WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    return row["template"] if row else None


def set_prompt(name: str, template: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO prompts (name,template) VALUES (?,?) "
        "ON CONFLICT(name) DO UPDATE SET template=excluded.template",
        (name, template),
    )
    conn.commit()
    conn.close()


# ── Feedback ──────────────────────────────────────────────────────────────────

def insert_feedback(email_id: str, field: str, old_value: str, new_value: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (email_id,field,old_value,new_value,corrected_at) "
        "VALUES (?,?,?,?,?)",
        (email_id, field, str(old_value), str(new_value), utcnow_iso()),
    )
    conn.commit()
    conn.close()


def get_all_feedback() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY corrected_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_feedback_preferences() -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY corrected_at ASC"
    ).fetchall()
    conn.close()

    sender_overrides       = {}
    low_priority_senders   = set()
    category_corrections   = {}
    confidence_adjustments = {}
    wrong_task_patterns    = []

    for row in rows:
        field     = row["field"]
        new_value = row["new_value"]
        old_value = row["old_value"]
        email_id  = row["email_id"]
        email     = get_email(email_id)
        sender    = email["sender"] if email else None

        if field == "category":
            if sender:
                sender_overrides[sender] = new_value
            if old_value:
                category_corrections[old_value] = new_value
            confidence_adjustments[new_value] = (
                confidence_adjustments.get(new_value, 0) + 0.05
            )
            if old_value:
                confidence_adjustments[old_value] = (
                    confidence_adjustments.get(old_value, 0) - 0.05
                )
        elif field == "priority" and new_value in ("6","7") and sender:
            low_priority_senders.add(sender)
        elif field == "task":
            if old_value and new_value:
                wrong_task_patterns.append((old_value, new_value))

    return {
        "sender_category_overrides": sender_overrides,
        "low_priority_senders":      list(low_priority_senders),
        "category_corrections":      category_corrections,
        "confidence_adjustments":    confidence_adjustments,
        "wrong_task_patterns":       wrong_task_patterns,
    }


# ── Jobs ──────────────────────────────────────────────────────────────────────

def create_job(job_id: str, job_type: str, total: int = 0) -> dict:
    now  = utcnow_iso()
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO jobs "
        "(id,job_type,status,total,completed,errors,progress,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (job_id, job_type, "pending", total, 0, 0, 0, now, now),
    )
    conn.commit()
    conn.close()
    return get_job(job_id)


def update_job(job_id: str, **kwargs):
    if not kwargs:
        return
    allowed = {
        "status","total","completed","errors","progress",
        "result_json","error_msg","finished_at",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    fields["updated_at"] = utcnow_iso()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values     = list(fields.values()) + [job_id]
    conn       = get_connection()
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", values)
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if d.get("result_json"):
        try:
            d["results"] = json.loads(d["result_json"])
        except Exception:
            d["results"] = []
    return d


def get_recent_jobs(limit: int = 10) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── ML training data ──────────────────────────────────────────────────────────

def save_training_sample(
    email_id: str, features: list[float], label: int, source: str = "feedback"
):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO ml_training_data "
        "(email_id,features_json,label,source,created_at) VALUES (?,?,?,?,?)",
        (email_id, json.dumps(features), label, source, utcnow_iso()),
    )
    conn.commit()
    conn.close()


def get_all_training_data() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM ml_training_data ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        try:
            d["features"] = json.loads(d["features_json"])
        except Exception:
            d["features"] = []
        result.append(d)
    return result


def save_model_version(version: str, accuracy: float, n_samples: int, model_path: str):
    conn = get_connection()
    conn.execute("UPDATE ml_models SET is_active=0")
    conn.execute(
        "INSERT INTO ml_models "
        "(version,accuracy,n_samples,model_path,is_active,trained_at) "
        "VALUES (?,?,?,?,1,?)",
        (version, accuracy, n_samples, model_path, utcnow_iso()),
    )
    conn.commit()
    conn.close()


def get_active_model_version() -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM ml_models WHERE is_active=1 "
        "ORDER BY trained_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_model_history() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM ml_models ORDER BY trained_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Rate limiting (DB fallback when Redis unavailable) ───────────────────────

def check_rate_limit(
    user_id: str, action: str, max_count: int, window_minutes: int
) -> tuple[bool, int]:
    from datetime import datetime, timedelta
    now    = datetime.utcnow()
    window = (now - timedelta(minutes=window_minutes)).isoformat()
    conn   = get_connection()
    row    = conn.execute(
        "SELECT count, window_start FROM rate_limits "
        "WHERE user_id=? AND action=?",
        (user_id, action),
    ).fetchone()

    if not row or row["window_start"] < window:
        conn.execute(
            "INSERT OR REPLACE INTO rate_limits "
            "(user_id,action,count,window_start) VALUES (?,?,1,?)",
            (user_id, action, now.isoformat()),
        )
        conn.commit()
        conn.close()
        return True, max_count - 1

    count = row["count"]
    if count >= max_count:
        conn.close()
        return False, 0

    conn.execute(
        "UPDATE rate_limits SET count=count+1 "
        "WHERE user_id=? AND action=?",
        (user_id, action),
    )
    conn.commit()
    conn.close()
    return True, max_count - count - 1


# ── Metrics ───────────────────────────────────────────────────────────────────

def log_metric(
    tool: str, email_id: str, latency_ms: float,
    success: bool, error_msg: str = ""
):
    conn = get_connection()
    conn.execute(
        "INSERT INTO metrics "
        "(tool,email_id,latency_ms,success,error_msg,called_at) "
        "VALUES (?,?,?,?,?,?)",
        (tool, email_id, round(latency_ms, 2),
         int(success), error_msg, utcnow_iso()),
    )
    conn.commit()
    conn.close()


def get_metrics_summary() -> dict:
    conn = get_connection()
    rows = conn.execute("""
        SELECT tool,
               COUNT(*) AS calls,
               ROUND(AVG(latency_ms),1) AS avg_latency_ms,
               SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS errors
        FROM metrics GROUP BY tool ORDER BY calls DESC
    """).fetchall()
    conn.close()
    return {r["tool"]: dict(r) for r in rows}


def get_total_llm_calls() -> int:
    conn = get_connection()
    row  = conn.execute(
        "SELECT COUNT(*) AS total FROM metrics WHERE success=1"
    ).fetchone()
    conn.close()
    return row["total"] if row else 0