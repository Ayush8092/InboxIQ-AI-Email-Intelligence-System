import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logger
from config.constants import MAX_RETRIES
from memory.repository import log_metric, upsert_processed, get_processed
from tools.categorize import categorize_email
from tools.extract_tasks import extract_tasks
from tools.priority import compute_priority
from tools.summarize import summarize_email
from tools.reply import generate_reply
from tools.snooze import snooze_email

logger = setup_logger(__name__)

TOOL_MAP = {
    "categorize_email": categorize_email,
    "extract_tasks":    extract_tasks,
    "compute_priority": compute_priority,
    "summarize_email":  summarize_email,
    "generate_reply":   generate_reply,
    "snooze_email":     snooze_email,
}

PARALLEL_SAFE = {"extract_tasks", "summarize_email"}
_HIGH_RISK     = {"Alert / Urgent", "Action Required", "Billing / Invoice"}


def _compute_needs_review(processed: dict) -> tuple[bool, list[str]]:
    reasons  = []
    cat_conf = processed.get("confidence")
    task     = processed.get("task")
    category = processed.get("category")

    if not category:
        reasons.append("Category could not be determined")
        return True, reasons

    if cat_conf is not None and float(cat_conf) < 0.50:
        reasons.append(
            f"Very low categorization confidence ({float(cat_conf):.0%})"
        )
    elif cat_conf is not None and float(cat_conf) < 0.60:
        task_weak = task is not None and len(task.split()) < 4
        if task_weak:
            reasons.append(
                f"Low confidence ({float(cat_conf):.0%}) with incomplete task"
            )

    if category in _HIGH_RISK and cat_conf is not None and float(cat_conf) < 0.70:
        r = (
            f"High-risk category '{category}' with moderate confidence "
            f"({float(cat_conf):.0%}) — verify before acting"
        )
        if r not in reasons:
            reasons.append(r)

    return len(reasons) > 0, reasons


def run_tool_with_retry(tool_name: str, email_id: str, **kwargs) -> dict:
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool: {tool_name}"}

    for attempt in range(MAX_RETRIES + 1):
        start = time.time()
        try:
            result  = fn(email_id, **kwargs)
            latency = (time.time() - start) * 1000
            log_metric(tool_name, email_id, latency, success=True)
            logger.info(
                f"Tool '{tool_name}' ok | {email_id} latency={latency:.0f}ms"
            )
            return result
        except Exception as e:
            latency = (time.time() - start) * 1000
            log_metric(tool_name, email_id, latency, success=False, error_msg=str(e))
            logger.warning(
                f"Tool '{tool_name}' attempt {attempt+1} failed | {e}"
            )
            if attempt == MAX_RETRIES:
                return {"error": str(e), "tool": tool_name}


def execute_plan(email: dict, plan: dict, persona: str = "Formal") -> dict:
    email_id  = email["id"]
    processed = get_processed(email_id) or {
        "email_id":      email_id,
        "category":      None,
        "priority":      3,
        "task":          None,
        "task_type":     "task",
        "steps":         None,
        "deadline":      None,
        "summary":       None,
        "confidence":    None,
        "needs_review":  0,
        "review_reason": "",
    }

    tool_results = {}
    tools        = list(plan.get("tools_to_call", []))

    # Step 1 — categorize first
    if "categorize_email" in tools:
        result = run_tool_with_retry("categorize_email", email_id)
        tool_results["categorize_email"] = result
        if "error" not in result:
            processed["category"]   = result.get("category")
            processed["confidence"] = result.get("confidence")

    # Step 2 — early review check before reply
    needs_review_early, reasons_early = _compute_needs_review(processed)
    if needs_review_early and "generate_reply" in tools:
        tools.remove("generate_reply")
        plan.setdefault("skip_reasons", {})["generate_reply"] = (
            "Skipped — needs human review: " + "; ".join(reasons_early)
        )

    # Step 3 — parallel tools
    parallel_tools = [t for t in tools if t in PARALLEL_SAFE]
    if parallel_tools:
        with ThreadPoolExecutor(max_workers=len(parallel_tools)) as executor:
            futures = {
                executor.submit(run_tool_with_retry, t, email_id): t
                for t in parallel_tools
            }
            for future in as_completed(futures):
                tool_name = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"error": str(e), "tool": tool_name}
                tool_results[tool_name] = result

                if "error" not in result and tool_name == "extract_tasks":
                    processed["task"]      = result.get("task")
                    processed["task_type"] = result.get("task_type", "task")
                    processed["steps"]     = result.get("steps")
                    processed["deadline"]  = result.get("deadline")

    # Step 4 — remaining sequential
    remaining = [
        t for t in tools
        if t not in PARALLEL_SAFE and t != "categorize_email"
    ]
    for tool in remaining:
        kwargs = {"persona": persona} if tool == "generate_reply" else {}
        result = run_tool_with_retry(tool, email_id, **kwargs)
        tool_results[tool] = result
        if "error" not in result:
            if tool == "compute_priority":
                processed["priority"] = result.get("priority")
            elif tool == "summarize_email":
                processed["summary"]  = result.get("summary")

    # Step 5 — final needs_review
    needs_review_final, reasons_final = _compute_needs_review(processed)
    if plan.get("needs_review") and "Planner flagged" not in " ".join(reasons_final):
        needs_review_final = True
        reasons_final.append("Planner flagged for review")

    processed["needs_review"]  = int(needs_review_final)
    processed["review_reason"] = "; ".join(reasons_final) if reasons_final else ""

    upsert_processed(processed)
    logger.info(
        f"Execution done | {email_id} "
        f"type={processed.get('task_type')} "
        f"task={processed.get('task')!r} "
        f"steps={len(processed.get('steps') or [])} "
        f"needs_review={needs_review_final}"
    )

    return {
        "email_id":     email_id,
        "plan":         plan,
        "tool_results": tool_results,
        "final_state":  processed,
    }