from utils.logger import setup_logger
from agent.planner import run_planner
from agent.executor import execute_plan
from memory.repository import get_email, insert_email, get_processed
from config.constants import DEFAULT_PERSONA

logger = setup_logger(__name__)


class Orchestrator:

    def handle_email(
        self,
        email_id: str,
        user_command: str = "Handle this email",
        persona: str = DEFAULT_PERSONA,
        dry_run: bool = False,
    ) -> dict:

        logger.info(f"Orchestrator | email_id={email_id} dry_run={dry_run}")

        email = get_email(email_id)
        if not email:
            return {"error": f"Email '{email_id}' not found."}

        plan = run_planner(email, user_command=user_command)

        if dry_run:
            return {
                "email_id":     email_id,
                "subject":      email["subject"],
                "sender":       email["sender"],
                "plan":         plan,
                "tool_results": {},
                "final_state":  get_processed(email_id) or {},
                "dry_run":      True,
            }

        result = execute_plan(email, plan, persona=persona)
        return {
            "email_id":     email_id,
            "subject":      email["subject"],
            "sender":       email["sender"],
            "plan":         plan,
            "tool_results": result["tool_results"],
            "final_state":  result["final_state"],
            "dry_run":      False,
        }

    def handle_all_emails(
        self,
        emails: list[dict],
        user_command: str = "Handle this email",
        persona: str = DEFAULT_PERSONA,
        dry_run: bool = False,
    ) -> list[dict]:
        results = []
        for email in emails:
            insert_email(email)
            results.append(self.handle_email(email["id"], user_command, persona, dry_run))
        return results


orchestrator = Orchestrator()