"""
Simple in-memory cache for LLM responses.
Prevents redundant API calls for identical prompts.
Reduces latency by up to 80% on repeated processing.
"""
import hashlib
import time
from utils.logger import setup_logger
from config.constants import CACHE_TTL_SECONDS, ENABLE_CACHE

logger = setup_logger(__name__)

_cache: dict[str, tuple[str, float]] = {}   # key → (response, timestamp)
_hits   = 0
_misses = 0


def _make_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def get_cached(prompt: str) -> str | None:
    global _hits, _misses
    if not ENABLE_CACHE:
        _misses += 1
        return None

    key = _make_key(prompt)
    if key in _cache:
        response, ts = _cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            _hits += 1
            logger.debug(f"Cache HIT | hits={_hits} misses={_misses}")
            return response
        else:
            del _cache[key]

    _misses += 1
    return None


def set_cached(prompt: str, response: str):
    if not ENABLE_CACHE or not response:
        return
    key          = _make_key(prompt)
    _cache[key]  = (response, time.time())


def cache_stats() -> dict:
    total = _hits + _misses
    return {
        "hits":       _hits,
        "misses":     _misses,
        "total":      total,
        "hit_rate":   round(_hits / total * 100, 1) if total > 0 else 0.0,
        "cache_size": len(_cache),
    }


def clear_cache():
    global _cache, _hits, _misses
    _cache  = {}
    _hits   = 0
    _misses = 0
    logger.info("Cache cleared.")