"""
LLM client using Redis for response caching.
Replaces in-memory cache with Redis for persistence across restarts.
"""
import time
from utils.secure_logger import get_secure_logger
from utils.redis_client import get_llm_cache, set_llm_cache
from config.config import GROQ_MODEL

logger  = get_secure_logger(__name__)
_client = None


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        from services.secrets_manager import get_groq_api_key
        _client = Groq(api_key=get_groq_api_key())
    return _client


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int    = 512,
    use_cache: bool    = True,
    timeout: int       = 30,
    max_retries: int   = 3,
) -> str:
    """LLM call with Redis caching and exponential backoff retry."""
    from config.config import GROQ_MODEL

    # Cache deterministic (temp=0) calls only
    should_cache = use_cache and temperature == 0.0

    if should_cache:
        cached = get_llm_cache(prompt)
        if cached:
            logger.debug("LLM Redis cache hit")
            from utils.observability import record_cache_hit
            record_cache_hit("llm_redis")
            return cached

    client     = get_client()
    last_error = None

    for attempt in range(max_retries):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text    = response.choices[0].message.content.strip()
            latency = time.time() - start

            logger.debug(
                f"LLM ok | attempt={attempt+1} "
                f"latency={latency*1000:.0f}ms chars={len(text)}"
            )

            from utils.observability import record_llm_call
            record_llm_call("groq", latency, "success")

            if should_cache and text:
                set_llm_cache(prompt, text)

            return text

        except Exception as e:
            last_error = e
            latency    = time.time() - start
            wait       = 2 ** attempt

            from utils.observability import record_llm_call
            record_llm_call("groq", latency, "error")

            logger.warning(
                f"LLM fail | attempt={attempt+1} "
                f"retrying in {wait}s | error={type(e).__name__}"
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.error(f"LLM total failure: {type(last_error).__name__}")
    return ""