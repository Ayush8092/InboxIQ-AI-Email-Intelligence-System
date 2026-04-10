"""
Redis client with full graceful fallback.
On Render free tier Redis is not available.
All functions return safe defaults when Redis is down.
"""
import os
import json
import hashlib
import time
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

REDIS_URL         = os.getenv("REDIS_URL", "")
_client           = None
_redis_available  = None   # cached availability check


def get_redis():
    global _client, _redis_available
    if _redis_available is False:
        return None
    if _client is not None:
        return _client
    if not REDIS_URL:
        _redis_available = False
        return None
    try:
        import redis
        c = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        c.ping()
        _client          = c
        _redis_available = True
        logger.info("Redis connected")
        return _client
    except Exception:
        _redis_available = False
        logger.info("Redis not available — using in-memory fallback")
        return None


def is_redis_available() -> bool:
    return get_redis() is not None


# ── In-memory fallbacks (used when Redis is absent) ───────────────────────────

_memory_cache:      dict = {}
_processed_ids:     set  = set()
_rate_counters:     dict = {}
_job_statuses:      dict = {}


# ── Idempotency ────────────────────────────────────────────────────────────────

def mark_email_processed(email_id: str, result_summary: dict | None = None) -> bool:
    r = get_redis()
    if r:
        try:
            key    = f"processed:{email_id}"
            result = r.set(key, json.dumps(result_summary or {}), ex=86400, nx=True)
            return result is not None
        except Exception:
            pass
    # In-memory fallback
    if email_id in _processed_ids:
        return False
    _processed_ids.add(email_id)
    return True


def is_email_processed(email_id: str) -> bool:
    r = get_redis()
    if r:
        try:
            return r.exists(f"processed:{email_id}") > 0
        except Exception:
            pass
    return email_id in _processed_ids


def clear_idempotency(email_id: str):
    r = get_redis()
    if r:
        try:
            r.delete(f"processed:{email_id}")
        except Exception:
            pass
    _processed_ids.discard(email_id)


# ── Rate limiting ──────────────────────────────────────────────────────────────

def check_rate_limit_redis(
    user_id: str,
    action: str,
    max_count: int,
    window_seconds: int,
) -> tuple[bool, int]:
    r = get_redis()
    if r:
        try:
            key = f"rate:{user_id}:{action}"
            now = time.time()
            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window_seconds + 1)
            results   = pipe.execute()
            count     = results[1]
            if count >= max_count:
                return False, 0
            return True, max_count - count - 1
        except Exception:
            pass
    # In-memory fallback
    key = f"{user_id}:{action}"
    now = time.time()
    if key not in _rate_counters:
        _rate_counters[key] = []
    _rate_counters[key] = [
        t for t in _rate_counters[key]
        if now - t < window_seconds
    ]
    if len(_rate_counters[key]) >= max_count:
        return False, 0
    _rate_counters[key].append(now)
    return True, max_count - len(_rate_counters[key])


# ── LLM caching ───────────────────────────────────────────────────────────────

CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))


def get_llm_cache(prompt: str) -> str | None:
    r = get_redis()
    key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    if r:
        try:
            return r.get(key)
        except Exception:
            pass
    return _memory_cache.get(key)


def set_llm_cache(prompt: str, response: str):
    key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    r   = get_redis()
    if r:
        try:
            r.set(key, response, ex=CACHE_TTL)
            return
        except Exception:
            pass
    # In-memory: cap at 500 entries
    if len(_memory_cache) >= 500:
        oldest = next(iter(_memory_cache))
        del _memory_cache[oldest]
    _memory_cache[key] = response


def get_cache_stats() -> dict:
    r = get_redis()
    if r:
        try:
            info = r.info("stats")
            return {
                "available": True,
                "hits":      info.get("keyspace_hits", 0),
                "misses":    info.get("keyspace_misses", 0),
                "hit_rate":  round(
                    info.get("keyspace_hits", 0) /
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1) * 100,
                    1
                ),
            }
        except Exception:
            pass
    return {
        "available": False,
        "hits":      0,
        "misses":    0,
        "hit_rate":  0,
        "note":      "Using in-memory cache",
        "cached_prompts": len(_memory_cache),
    }


# ── Job status ────────────────────────────────────────────────────────────────

def set_job_status(job_id: str, status: dict):
    r = get_redis()
    if r:
        try:
            r.set(f"job:{job_id}", json.dumps(status), ex=86400)
            return
        except Exception:
            pass
    _job_statuses[job_id] = status


def get_job_status_redis(job_id: str) -> dict | None:
    r = get_redis()
    if r:
        try:
            val = r.get(f"job:{job_id}")
            return json.loads(val) if val else None
        except Exception:
            pass
    return _job_statuses.get(job_id)


def publish_job_progress(job_id: str, progress: int, message: str = ""):
    set_job_status(job_id, {
        "progress": progress,
        "message":  message,
        "ts":       time.time(),
    })