"""
Production-grade LLM service.
Features:
- Exponential backoff retry
- Timeout handling
- Output schema validation
- Response caching
- Token usage optimization
- No sensitive data in logs
"""
import time
import json
import re
from utils.secure_logger import get_secure_logger
from utils.cache import get_cached, set_cached

logger = get_secure_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        from config.config import GROQ_API_KEY
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _extract_json(text: str) -> str:
    """Extract JSON object from text that may contain surrounding prose."""
    if not text:
        return ""
    # Strip markdown
    cleaned = (
        text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
    )
    # Find first JSON object
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    # Fix unescaped newlines inside strings
    result      = []
    in_string   = False
    escape_next = False
    for char in cleaned:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\r':
            result.append('\\r')
        elif in_string and char == '\t':
            result.append('\\t')
        else:
            result.append(char)
    return ''.join(result)


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int    = 512,
    use_cache: bool    = True,
    timeout: int       = 30,
    max_retries: int   = 3,
) -> str:
    """
    Production LLM call with:
    - Response caching (deterministic calls only)
    - Exponential backoff retry
    - Timeout handling
    - Never logs prompt content (may contain email data)
    """
    from config.config import GROQ_MODEL

    # Cache deterministic calls
    if use_cache and temperature == 0.0:
        cached = get_cached(prompt)
        if cached is not None:
            logger.debug("LLM cache hit")
            return cached

    client     = _get_client()
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
            latency = (time.time() - start) * 1000
            logger.debug(
                f"LLM ok | attempt={attempt+1} latency={latency:.0f}ms "
                f"tokens={len(text.split())}"
            )

            if use_cache and temperature == 0.0 and text:
                set_cached(prompt, text)

            return text

        except Exception as e:
            last_error = e
            latency    = (time.time() - start) * 1000
            wait       = 2 ** attempt   # exponential backoff: 1s, 2s, 4s
            logger.warning(
                f"LLM fail | attempt={attempt+1} latency={latency:.0f}ms "
                f"retrying in {wait}s | error={type(e).__name__}"
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.error(f"LLM total failure after {max_retries} attempts: {type(last_error).__name__}")
    return ""


def call_llm_json(
    prompt: str,
    required_keys: list[str],
    fallback: dict,
    temperature: float = 0.0,
    max_tokens: int    = 300,
    use_cache: bool    = True,
) -> dict:
    """
    LLM call that guarantees structured JSON output.
    Validates required keys and returns fallback on schema failure.
    Never returns hallucinated fields.
    """
    raw = call_llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
    if not raw:
        return fallback

    # Extract and parse JSON
    cleaned = _extract_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | preview={raw[:80]!r}")
        return fallback

    # Validate required keys
    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.warning(f"LLM output missing required keys: {missing}")
        # Merge with fallback for missing keys
        for k in missing:
            data[k] = fallback.get(k)

    # Prevent hallucinated extra fields — only keep known keys
    allowed = set(fallback.keys())
    data    = {k: v for k, v in data.items() if k in allowed}

    return data