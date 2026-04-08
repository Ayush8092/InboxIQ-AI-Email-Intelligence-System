from memory.db import init_db, get_connection
from memory.repository import (
    insert_email, get_email, get_all_emails,
    upsert_processed, get_processed, get_all_processed, update_processed_field,
    insert_draft, get_drafts, get_all_drafts,
    get_prompt, set_prompt,
    insert_feedback, get_all_feedback,
    log_metric, get_metrics_summary, get_total_llm_calls,
    get_feedback_preferences,
)