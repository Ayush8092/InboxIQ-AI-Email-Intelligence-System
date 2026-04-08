"""
Redis client — single connection pool for all Redis operations.
Responsibilities:
  - Idempotency keys (email processing deduplication)
  - Rate limiting (per user/IP)
  - LLM response caching (replaces in-memory cache)
  - Celery broker (configured in celery_app.py)

NEVER used as primary database.
"""
import os
import json
import hashlib
import time
from typing import Any
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

# ── Connection ────────────────────────────────────────────────────────────────

_pool   = None
_client = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis():
    """
    Get thread-safe Redis client using connection pool.
    Returns None gracefully if Redis is unavailable.
    """
    global _pool, _client
    if _client is not None:
        return _client
    try:
        import redis
        _pool   = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        _client = redis.Redis(connection_pool=_pool)
        _client.ping()
        logger.info(f"Redis connected: {REDIS_URL.split('@')[-1]}")
        return _client
    except Exception as e:
        logger.warning(f"Redis unavailable: {type(e).__name__} — using fallback")
        return None


def is_redis_available() -> bool:
    """Check if Redis is reachable."""
    r = get_redis()
    if r is None:
        return False
    try:
        return r.ping()
    except Exception:
        return False


# ── Idempotency ───────────────────────────────────────────────────────────────

IDEMPOTENCY_TTL = 86400   # 24 hours


def mark_email_processed(email_id: str, result_summary: dict | None = None) -> bool:
    """
    Mark email as processed in Redis for idempotency.
    Returns True if successfully marked, False if already exists.
    """
    r = get_redis()
    if r is None:
        return True   # allow processing if Redis is down

    key   = f"processed:{email_id}"
    value = json.dumps(result_summary or {"ts": time.time()})

    # SET NX (set if not exists) — atomic idempotency check
    result = r.set(key, value, ex=IDEMPOTENCY_TTL, nx=True)
    if result is None:
        logger.debug(f"Idempotency hit — email already processed: {email_id}")
        return False
    return True


def is_email_processed(email_id: str) -> bool:
    """Check if email has already been processed."""
    r = get_redis()
    if r is None:
        return False
    try:
        return r.exists(f"processed:{email_id}") > 0
    except Exception:
        return False


def clear_idempotency(email_id: str):
    """Clear idempotency key to allow reprocessing."""
    r = get_redis()
    if r:
        r.delete(f"processed:{email_id}")


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit_redis(
    user_id: str,
    action: str,
    max_count: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Sliding window rate limiter using Redis sorted sets.
    More accurate than fixed window approach.

    Returns (allowed: bool, remaining: int).
    """
    r = get_redis()
    if r is None:
        return True, max_count   # allow if Redis is down

    key = f"rate:{user_id}:{action}"
    now = time.time()

    try:
        pipe = r.pipeline()
        # Remove entries outside the window
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        # Count current entries
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set key expiry
        pipe.expire(key, window_seconds + 1)
        results  = pipe.execute()
        count    = results[1]   # count BEFORE adding current request

        if count >= max_count:
            logger.warning(f"Rate limit exceeded | user={user_id} action={action}")
            return False, 0

        remaining = max_count - count - 1
        return True, remaining

    except Exception as e:
        logger.warning(f"Rate limit check failed: {type(e).__name__} — allowing")
        return True, max_count


# ── LLM Response Caching ──────────────────────────────────────────────────────

CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))   # 1 hour default


def get_llm_cache(prompt: str) -> str | None:
    """Get cached LLM response from Redis."""
    r = get_redis()
    if r is None:
        return None
    try:
        key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
        return r.get(key)
    except Exception:
        return None


def set_llm_cache(prompt: str, response: str):
    """Cache LLM response in Redis with TTL."""
    r = get_redis()
    if r is None:
        return
    try:
        key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
        r.set(key, response, ex=CACHE_TTL)
    except Exception as e:
        logger.debug(f"Cache set failed: {type(e).__name__}")


def get_cache_stats() -> dict:
    """Return Redis cache statistics."""
    r = get_redis()
    if r is None:
        return {"available": False}
    try:
        info      = r.info("stats")
        keyspace  = r.info("keyspace")
        llm_keys  = len(r.keys("llm:*"))
        proc_keys = len(r.keys("processed:*"))
        rate_keys = len(r.keys("rate:*"))
        return {
            "available":       True,
            "hits":            info.get("keyspace_hits", 0),
            "misses":          info.get("keyspace_misses", 0),
            "hit_rate":        round(
                info.get("keyspace_hits", 0) /
                max(info.get("keyspace_hits",0) + info.get("keyspace_misses",1), 1) * 100,
                1
            ),
            "llm_cache_keys":  llm_keys,
            "idempotency_keys": proc_keys,
            "rate_limit_keys": rate_keys,
            "total_keys":      sum(
                v.get("keys",0)
                for v in keyspace.values()
                if isinstance(v, dict)
            ),
        }
    except Exception as e:
        return {"available": True, "error": str(type(e).__name__)}


# ── Job status in Redis ────────────────────────────────────────────────────────

JOB_TTL = 3600 * 24   # 24 hours


def set_job_status(job_id: str, status: dict):
    """Store Celery job status in Redis."""
    r = get_redis()
    if r is None:
        return
    try:
        r.set(f"job:{job_id}", json.dumps(status), ex=JOB_TTL)
    except Exception:
        pass


def get_job_status_redis(job_id: str) -> dict | None:
    """Get job status from Redis."""
    r = get_redis()
    if r is None:
        return None
    try:
        val = r.get(f"job:{job_id}")
        return json.loads(val) if val else None
    except Exception:
        return None


def publish_job_progress(job_id: str, progress: int, message: str = ""):
    """Publish job progress to Redis pub/sub channel."""
    r = get_redis()
    if r is None:
        return
    try:
        r.publish(
            f"job_progress:{job_id}",
            json.dumps({"progress": progress, "message": message, "ts": time.time()})
        )
    except Exception:
        pass