import json
from utils.llm_client import call_llm
from utils.validators import parse_json_strict, validate_planner_output
from utils.logger import setup_logger
from agent.prompts import DEFAULT_PROMPTS
from memory.repository import get_prompt, get_processed, get_feedback_preferences
from config.constants import CONFIDENCE_HIGH, SKIP_REPLY_CATEGORIES, SKIP_TASK_CATEGORIES, NO_PROCESS_CATEGORIES

logger = setup_logger(__name__)


def _get_template(name: str) -> str:
    saved = get_prompt(name)
    return saved if saved else DEFAULT_PROMPTS[name]


def _enforce_skip_rules(plan: dict, state: dict, email: dict, prefs: dict) -> dict:
    """Enforce tool-skipping rules in Python — LLM cannot override these."""
    tools        = list(plan.get("tools_to_call", []))
    skip_reasons = dict(plan.get("skip_reasons", {}))
    needs_review = plan.get("needs_review", False)
    category     = state.get("category")
    confidence   = state.get("confidence")
    sender       = email.get("sender", "")

    # Apply feedback sender overrides
    sender_overrides = prefs.get("sender_category_overrides", {})
    if sender in sender_overrides:
        state["category"] = sender_overrides[sender]
        category          = state["category"]

    # Skip categorize if already high confidence
    if confidence and float(confidence) >= CONFIDENCE_HIGH and "categorize_email" in tools:
        tools.remove("categorize_email")
        skip_reasons["categorize_email"] = f"Already categorized with confidence {confidence:.0%}"

    # Skip reply for no-reply categories
    if category in SKIP_REPLY_CATEGORIES and "generate_reply" in tools:
        tools.remove("generate_reply")
        skip_reasons["generate_reply"] = f"Category '{category}' does not require a reply"

    # Skip task extraction for irrelevant categories
    if category in SKIP_TASK_CATEGORIES and "extract_tasks" in tools:
        tools.remove("extract_tasks")
        skip_reasons["extract_tasks"] = f"Category '{category}' has no actionable tasks"

    # For pure noise categories strip most tools
    if category in NO_PROCESS_CATEGORIES:
        noisy = {"extract_tasks","generate_reply","snooze_email"}
        removed = [t for t in tools if t in noisy]
        tools   = [t for t in tools if t not in noisy]
        for t in removed:
            skip_reasons[t] = f"Not needed for '{category}' emails"

    # Low-priority sender — skip reply
    low_senders = set(prefs.get("low_priority_senders", []))
    if sender in low_senders and "generate_reply" in tools:
        tools.remove("generate_reply")
        skip_reasons["generate_reply"] = f"Sender previously marked low-priority"

    # Low confidence → flag review
    if confidence and float(confidence) < 0.50:
        needs_review = True

    plan["tools_to_call"] = tools
    plan["skip_reasons"]  = skip_reasons
    plan["needs_review"]  = needs_review
    return plan


def run_planner(email: dict, user_command: str = "Handle this email") -> dict:
    current_state = get_processed(email["id"]) or {}
    prefs         = get_feedback_preferences()

    prompt = _get_template("planner").format(
        email_id=email["id"],
        subject=email["subject"],
        sender=email["sender"],
        current_state=json.dumps(current_state),
        user_preferences=json.dumps(prefs),
        user_command=user_command,
    )

    logger.info(f"Planner | email_id={email['id']}")
    raw = call_llm(prompt, temperature=0.0, max_tokens=300)

    fallback = {
        "tools_to_call": ["categorize_email","extract_tasks","compute_priority","summarize_email"],
        "skip_reasons":  {},
        "needs_review":  False,
        "explanation":   "Planner fallback — using default tool set.",
    }

    plan = parse_json_strict(raw, fallback=fallback, context="planner")

    if not validate_planner_output(plan):
        logger.warning(f"Planner validation failed for {email['id']}, using fallback.")
        plan = fallback

    plan = _enforce_skip_rules(plan, current_state, email, prefs)

    logger.info(
        f"Plan | email_id={email['id']} "
        f"tools={plan['tools_to_call']} "
        f"skipped={list(plan['skip_reasons'].keys())}"
    )
    return plan