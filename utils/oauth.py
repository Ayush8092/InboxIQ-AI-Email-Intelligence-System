"""
Gmail OAuth with:
- Source tagging (source='gmail')
- Email count parameter
- Optional OCR on image attachments
"""
import re
import secrets
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from utils.secure_logger import get_secure_logger
from utils.encryption import encrypt_token, decrypt_token
from utils.helpers import utcnow_iso

logger = get_secure_logger(__name__)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"
GMAIL_API_URL    = "https://gmail.googleapis.com/gmail/v1"
GMAIL_BATCH_URL  = "https://www.googleapis.com/batch/gmail/v1"

SAFE_BATCH_SIZE  = 50
BATCH_WORKERS    = 3
REQUEST_TIMEOUT  = 30
MAX_RETRIES      = 3
RETRY_BACKOFF    = [1, 2, 4]


def generate_oauth_state() -> str:
    return secrets.token_urlsafe(32)


def validate_oauth_state(received: str, expected: str) -> bool:
    if not received or not expected:
        logger.warning("OAuth state validation failed — missing state")
        return False
    return secrets.compare_digest(received, expected)


def get_auth_url(
    client_id: str,
    redirect_uri: str,
    state: str = "",
) -> str:
    params = {
        "client_id":              client_id,
        "redirect_uri":           redirect_uri,
        "response_type":          "code",
        "scope":                  " ".join(SCOPES),
        "access_type":            "offline",
        "include_granted_scopes": "true",
        "prompt":                 "consent",
    }
    if state:
        params["state"] = state
    q = "&".join(
        f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()
    )
    return f"{GOOGLE_AUTH_URL}?{q}"


def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict | None:
    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     client_id,
                "client_secret": client_secret,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.error(
                f"Token exchange HTTP {resp.status_code}: {resp.text}"
            )
            return None
        logger.info("Token exchange successful")
        return resp.json()
    except Exception as e:
        logger.error(f"Token exchange failed: {type(e).__name__}: {e}")
        return None


def store_tokens_in_session(session_state: dict, tokens: dict):
    session_state["enc_access_token"]  = encrypt_token(tokens.get("access_token",""))
    session_state["enc_refresh_token"] = encrypt_token(tokens.get("refresh_token",""))
    session_state["token_expiry"]      = time.time() + tokens.get("expires_in", 3600)
    logger.info("Tokens encrypted and stored")


def _do_refresh(enc_refresh: str, client_id: str, client_secret: str) -> str | None:
    rt = decrypt_token(enc_refresh)
    if not rt:
        return None
    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": rt,
                "client_id":     client_id,
                "client_secret": client_secret,
                "grant_type":    "refresh_token",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        tok = resp.json().get("access_token","")
        if tok:
            logger.info("Token refreshed")
        return tok
    except Exception as e:
        logger.error(f"Token refresh failed: {type(e).__name__}")
        return None


def get_valid_access_token(session_state: dict) -> str | None:
    from config.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    expiry      = session_state.get("token_expiry", 0)
    enc_access  = session_state.get("enc_access_token","")
    enc_refresh = session_state.get("enc_refresh_token","")
    if enc_access and time.time() < (expiry - 60):
        return decrypt_token(enc_access)
    if enc_refresh:
        new_tok = _do_refresh(enc_refresh, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
        if new_tok:
            session_state["enc_access_token"] = encrypt_token(new_tok)
            session_state["token_expiry"]     = time.time() + 3600
            return new_tok
    logger.warning("No valid token available")
    return None


def get_user_info(access_token: str) -> dict | None:
    try:
        resp = requests.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"User info failed: {type(e).__name__}")
        return None


def _decode_body(payload: dict) -> str:
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body",{}).get("data","")
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
        data = payload.get("body",{}).get("data","")
        if data:
            body = base64.urlsafe_b64decode(
                data + "=="
            ).decode("utf-8", errors="replace")
    return body.strip()


def _parse_message(data: dict, run_ocr: bool = False) -> dict | None:
    """
    Parse Gmail message into AEOA email dict.
    Always tags source='gmail'.
    Optionally runs OCR on image attachments.
    """
    try:
        hdrs     = data.get("payload",{}).get("headers",[])
        get_h    = lambda n: next(
            (h["value"] for h in hdrs if h["name"].lower() == n.lower()), ""
        )
        subject      = get_h("Subject") or "(No Subject)"
        sender       = get_h("From") or "unknown@unknown.com"
        date_str     = get_h("Date") or ""
        body         = _decode_body(data.get("payload",{}))
        m            = re.search(r'<(.+?)>', sender)
        sender_email = m.group(1) if m else sender

        try:
            ts = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            ts = utcnow_iso()

        # Optional OCR on image attachments
        if run_ocr:
            try:
                from services.ocr_service import process_email_attachments_ocr
                body = process_email_attachments_ocr(
                    body,
                    gmail_payload=data.get("payload",{}),
                )
            except Exception as e:
                logger.warning(f"OCR failed for {data.get('id')}: {type(e).__name__}")

        return {
            "id":        f"gmail_{data.get('id','')}",
            "subject":   subject[:200],
            "body":      body[:3000],
            "sender":    sender_email,
            "timestamp": ts,
            "source":    "gmail",   # ← always tag as gmail
        }
    except Exception as e:
        logger.warning(f"Message parse failed: {type(e).__name__}")
        return None


def _fetch_single_with_retry(
    msg_id: str,
    headers: dict,
    run_ocr: bool = False,
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{GMAIL_API_URL}/users/me/messages/{msg_id}",
                headers=headers,
                params={"format": "full"},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 429:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                logger.warning(f"Rate limited on {msg_id}, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return _parse_message(r.json(), run_ocr=run_ocr)
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {msg_id} (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"Fetch failed {msg_id}: {type(e).__name__}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF[attempt])
    logger.error(f"All retries failed for {msg_id}")
    return None


def _fetch_individually_parallel(
    msg_ids: list[str],
    access_token: str,
    run_ocr: bool = False,
) -> list[dict]:
    headers = {"Authorization": f"Bearer {access_token}"}
    emails  = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_with_retry, mid, headers, run_ocr): mid
            for mid in msg_ids
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                emails.append(result)
    return emails


def fetch_gmail_emails(
    access_token: str,
    max_results: int = 20,
    run_ocr: bool = False,
) -> list[dict]:
    """
    Fetch Gmail emails.
    All returned emails have source='gmail'.

    max_results: 10-75 controlled from UI slider
    run_ocr:     if True, run Vision API OCR on image attachments
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    # Step 1: Get message IDs
    try:
        r = requests.get(
            f"{GMAIL_API_URL}/users/me/messages",
            headers=headers,
            params={"maxResults": max_results, "q": "in:inbox"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        messages = r.json().get("messages",[])
    except Exception as e:
        logger.error(f"Gmail list failed: {type(e).__name__}")
        return []

    if not messages:
        return []

    msg_ids = [m["id"] for m in messages]
    logger.info(f"Fetching {len(msg_ids)} Gmail messages (ocr={run_ocr})")

    # Step 2: Fetch in parallel
    emails = _fetch_individually_parallel(msg_ids, access_token, run_ocr)
    emails.sort(key=lambda x: x.get("timestamp",""), reverse=True)
    logger.info(f"Gmail fetch complete: {len(emails)} emails (source=gmail)")
    return emails