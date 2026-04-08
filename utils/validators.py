import json
import re
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _extract_json_from_text(raw: str) -> str:
    """
    Aggressively extract JSON from LLM output that may contain
    surrounding text, markdown fences, or explanations.
    """
    if not raw:
        return ""

    # Step 1: strip markdown fences
    cleaned = (
        raw.strip()
           .removeprefix("```json")
           .removeprefix("```")
           .removesuffix("```")
           .strip()
    )

    # Step 2: try to find JSON object using regex
    # Look for the first { ... } block
    json_pattern = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_pattern:
        cleaned = json_pattern.group(0)

    # Step 3: fix unescaped newlines/tabs inside string values
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


def parse_json_strict(raw: str, fallback: dict, context: str = "") -> dict:
    """
    Three-attempt JSON parsing:
    1. Direct parse
    2. Extract JSON block then parse
    3. Return fallback
    """
    if not raw:
        logger.warning(f"parse_json_strict [{context}]: empty raw string")
        return fallback

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract and clean
    cleaned = _extract_json_from_text(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON parse failed [{context}]: {e} | "
            f"raw_preview={raw[:150]!r}"
        )
        return fallback


def validate_category_output(data: dict) -> bool:
    """Lenient — accepts string confidence, checks category is non-empty string."""
    if not isinstance(data.get("category"), str):
        return False
    if not data["category"].strip():
        return False
    try:
        conf = float(data.get("confidence", -1))
        return 0.0 <= conf <= 1.0
    except (TypeError, ValueError):
        return False


def validate_task_output(data: dict) -> bool:
    return "task" in data and "deadline" in data and "confidence" in data


def validate_reply_output(data: dict) -> bool:
    return (
        isinstance(data.get("subject"), str) and
        isinstance(data.get("body"), str) and
        len(data["body"]) > 0
    )


def validate_planner_output(data: dict) -> bool:
    return (
        isinstance(data.get("tools_to_call"), list) and
        len(data["tools_to_call"]) > 0 and
        isinstance(data.get("explanation"), str)
    )