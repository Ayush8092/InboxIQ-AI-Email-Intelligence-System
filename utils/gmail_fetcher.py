"""
Production Gmail batch fetcher.
- Safe batch size (50 messages per batch)
- Per-request retry with exponential backoff
- Partial failure handling — failed sub-requests retried individually
- Structured logging per email failure
- Fallback to individual fetch on total batch failure
"""
import re
import uuid
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from utils.secure_logger import get_secure_logger
from utils.helpers import utcnow_iso

logger = get_secure_logger(__name__)

GMAIL_API_URL   = "https://gmail.googleapis.com/gmail/v1"
GMAIL_BATCH_URL = "https://www.googleapis.com/batch/gmail/v1"

# Production-safe constants
SAFE_BATCH_SIZE  = 50    # Google recommends ≤100, we use 50 for safety
BATCH_WORKERS    = 3     # parallel batch threads
REQUEST_TIMEOUT  = 30    # seconds per request
MAX_RETRIES      = 3     # retries per failed sub-request
RETRY_BACKOFF    = [1, 2, 4]  # exponential backoff seconds


def _decode_body(payload: dict) -> str:
    """Decode Gmail message body — plain text only."""
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    body = base64.urlsafe_b64decode(
                        data + "=="
                    ).decode("utf-8", errors="replace")
                    break
            elif "parts" in part:
                body = _decode_body(part)
                if body:
                    break
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            body = base64.urlsafe_b64decode(
                data + "=="
            ).decode("utf-8", errors="replace")
    return body.strip()


def _parse_message(data: dict) -> dict | None:
    """Parse Gmail message dict into AEOA email dict."""
    try:
        hdrs     = data.get("payload", {}).get("headers", [])
        get_h    = lambda n: next(
            (h["value"] for h in hdrs if h["name"].lower() == n.lower()), ""
        )
        subject      = get_h("Subject") or "(No Subject)"
        sender       = get_h("From") or "unknown@unknown.com"
        date_str     = get_h("Date") or ""
        body         = _decode_body(data.get("payload", {}))
        m            = re.search(r'<(.+?)>', sender)
        sender_email = m.group(1) if m else sender
        try:
            ts = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            ts = utcnow_iso()
        return {
            "id":        f"gmail_{data.get('id','')}",
            "subject":   subject[:200],
            "body":      body[:2000],
            "sender":    sender_email,
            "timestamp": ts,
        }
    except Exception as e:
        logger.warning(f"Message parse failed: {type(e).__name__}")
        return None


def _fetch_single_with_retry(
    msg_id: str,
    headers: dict,
    attempt: int = 0,
) -> dict | None:
    """
    Fetch a single Gmail message with retry + backoff.
    Used as fallback when batch sub-request fails.
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{GMAIL_API_URL}/users/me/messages/{msg_id}",
                headers=headers,
                params={"format": "full"},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 429:  # rate limited
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                logger.warning(f"Rate limited on {msg_id}, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return _parse_message(r.json())
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {msg_id} (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {msg_id}: {type(e).__name__}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF[attempt])
    logger.error(f"All retries failed for message {msg_id}")
    return None


def _build_batch_body(msg_ids: list[str]) -> tuple[str, str]:
    """Build multipart/mixed batch request body."""
    boundary = f"batch_{uuid.uuid4().hex}"
    parts    = []
    for msg_id in msg_ids:
        part = (
            f"--{boundary}\r\n"
            f"Content-Type: application/http\r\n"
            f"Content-ID: <item_{msg_id}>\r\n\r\n"
            f"GET /gmail/v1/users/me/messages/{msg_id}?format=full HTTP/1.1\r\n"
            f"Host: www.googleapis.com\r\n\r\n"
        )
        parts.append(part)
    body = "".join(parts) + f"--{boundary}--\r\n"
    return boundary, body


def _parse_batch_response(
    response_text: str,
    boundary: str,
) -> tuple[list[dict], list[str]]:
    """
    Parse multipart batch response.
    Returns (successful_messages, failed_msg_ids).
    Handles partial failures gracefully.
    """
    import json as _json
    successful = []
    failed_ids = []

    parts = response_text.split(f"--{boundary}")
    for part in parts:
        if "Content-Type: application/http" not in part:
            continue

        # Extract Content-ID to track which message this is
        cid_match = re.search(r'Content-ID:\s*<item_([^>]+)>', part)
        msg_id    = cid_match.group(1) if cid_match else None

        # Check HTTP status in sub-response
        status_match = re.search(r'HTTP/\d\.\d\s+(\d+)', part)
        status_code  = int(status_match.group(1)) if status_match else 200

        if status_code >= 400:
            logger.warning(
                f"Batch sub-request failed | msg_id={msg_id} "
                f"status={status_code}"
            )
            if msg_id:
                failed_ids.append(msg_id)
            continue

        # Extract JSON body
        json_match = re.search(r'\r\n\r\n(\{.*\})\s*$', part, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{"id".*\})\s*$', part, re.DOTALL)

        if json_match:
            try:
                data = _json.loads(json_match.group(1))
                if "id" in data and "payload" in data:
                    parsed = _parse_message(data)
                    if parsed:
                        successful.append(parsed)
                    else:
                        if msg_id:
                            failed_ids.append(msg_id)
                else:
                    if msg_id:
                        failed_ids.append(msg_id)
            except _json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed in batch part: {e}")
                if msg_id:
                    failed_ids.append(msg_id)
        else:
            if msg_id:
                failed_ids.append(msg_id)

    return successful, failed_ids


def _execute_batch(
    msg_ids: list[str],
    access_token: str,
) -> list[dict]:
    """
    Execute a single batch request.
    Retries failed sub-requests individually.
    Returns all successfully parsed emails.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    boundary, body = _build_batch_body(msg_ids)
    batch_headers  = {
        **headers,
        "Content-Type": f"multipart/mixed; boundary={boundary}",
    }

    emails     = []
    failed_ids = list(msg_ids)  # assume all failed until proven otherwise

    try:
        resp = requests.post(
            GMAIL_BATCH_URL,
            headers=batch_headers,
            data=body.encode("utf-8"),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 429:
            logger.warning("Batch rate limited — falling back to individual fetch")
            return _fetch_individually_parallel(msg_ids, access_token)

        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        m  = re.search(r'boundary=([^\s;]+)', ct)
        if not m:
            logger.warning("No boundary in batch response — falling back")
            return _fetch_individually_parallel(msg_ids, access_token)

        resp_boundary        = m.group(1).strip('"')
        successful, failed_ids = _parse_batch_response(resp.text, resp_boundary)
        emails.extend(successful)

        logger.info(
            f"Batch result | success={len(successful)} "
            f"failed={len(failed_ids)} batch_size={len(msg_ids)}"
        )

    except requests.exceptions.RequestException as e:
        logger.error(
            f"Batch request failed entirely: {type(e).__name__} — "
            f"falling back to individual fetch for {len(msg_ids)} messages"
        )
        return _fetch_individually_parallel(msg_ids, access_token)

    # Retry each failed sub-request individually with backoff
    if failed_ids:
        logger.info(f"Retrying {len(failed_ids)} failed sub-requests individually")
        for msg_id in failed_ids:
            result = _fetch_single_with_retry(msg_id, headers)
            if result:
                emails.append(result)
            else:
                logger.error(f"Final failure — could not fetch message {msg_id}")

    return emails


def _fetch_individually_parallel(
    msg_ids: list[str],
    access_token: str,
) -> list[dict]:
    """Fallback: fetch all messages individually in parallel."""
    headers = {"Authorization": f"Bearer {access_token}"}
    emails  = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_with_retry, mid, headers): mid
            for mid in msg_ids
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                emails.append(result)
    return emails


def fetch_gmail_emails(
    access_token: str,
    max_results: int = 50,
) -> list[dict]:
    """
    Production Gmail fetcher.
    - Safe batch size (SAFE_BATCH_SIZE)
    - Parallel batch execution
    - Per-sub-request retry
    - Partial failure recovery
    - Full fallback to individual fetching
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    # Step 1 — Get message ID list
    try:
        r = requests.get(
            f"{GMAIL_API_URL}/users/me/messages",
            headers=headers,
            params={"maxResults": max_results, "q": "in:inbox"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        messages = r.json().get("messages", [])
    except Exception as e:
        logger.error(f"Gmail list failed: {type(e).__name__}")
        return []

    if not messages:
        return []

    msg_ids = [m["id"] for m in messages]
    logger.info(
        f"Starting batch fetch | total={len(msg_ids)} "
        f"batch_size={SAFE_BATCH_SIZE} workers={BATCH_WORKERS}"
    )

    # Step 2 — Split into safe batches
    batches = [
        msg_ids[i:i + SAFE_BATCH_SIZE]
        for i in range(0, len(msg_ids), SAFE_BATCH_SIZE)
    ]

    # Step 3 — Execute batches in parallel
    all_emails = []
    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
        futures = {
            executor.submit(_execute_batch, batch, access_token): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=120)
                all_emails.extend(result)
            except Exception as e:
                logger.error(f"Batch worker exception: {type(e).__name__}")

    all_emails.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logger.info(
        f"Fetch complete | fetched={len(all_emails)} "
        f"requested={len(msg_ids)}"
    )
    return all_emails