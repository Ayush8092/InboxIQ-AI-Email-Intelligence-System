import json
import re
from typing import Optional, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parser that handles LLM quirks:
    - Extra text before/after JSON
    - Markdown code fences
    - Unescaped newlines inside strings
    """
    if not text:
        return None

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: strip markdown fences
    cleaned = (
        text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
    )
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Attempt 3: extract JSON block with regex
    try:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            candidate = match.group()
            return json.loads(candidate)
    except Exception:
        pass

    # Attempt 4: fix unescaped newlines inside strings then retry
    try:
        fixed = _fix_newlines_in_strings(cleaned)
        return json.loads(fixed)
    except Exception:
        pass

    # Attempt 5: extract JSON block after fixing newlines
    try:
        match = re.search(r'\{.*\}', _fix_newlines_in_strings(text), re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    logger.warning(f"safe_parse_json: all attempts failed | text={text[:150]}")
    return None


def _fix_newlines_in_strings(s: str) -> str:
    """Replace literal newlines inside JSON string values with \\n."""
    result = []
    in_string = False
    escape_next = False
    for char in s:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == '\n' and in_string:
            result.append('\\n')
        elif char == '\r' and in_string:
            result.append('\\r')
        elif char == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(char)
    return ''.join(result)


def extract_category_fallback(text: str) -> Optional[str]:
    """Extract category string even if full JSON parse fails."""
    match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def extract_confidence_fallback(text: str) -> float:
    """Extract confidence float even if full JSON parse fails."""
    match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.5