from utils.llm_client import call_llm
from utils.validators import parse_json_strict, validate_reply_output
from utils.logger import setup_logger
from utils.helpers import load_json_file
from agent.prompts import DEFAULT_PROMPTS
from memory.repository import get_email, get_prompt, insert_draft
from config.constants import DEFAULT_PERSONA

logger = setup_logger(__name__)


def _get_persona_desc(persona: str) -> str:
    try:
        personas = load_json_file("data/personas.json")
        return personas.get(persona, {}).get("description", "Professional tone.")
    except Exception:
        return "Professional tone."


def generate_reply(email_id: str, persona: str = DEFAULT_PERSONA) -> dict:
    email = get_email(email_id)
    if not email:
        raise ValueError(f"Email '{email_id}' not found.")

    persona_desc = _get_persona_desc(persona)
    template     = get_prompt("generate_reply") or DEFAULT_PROMPTS["generate_reply"]
    prompt       = template.format(
        subject=email["subject"],
        body=email["body"][:400],
        sender=email["sender"],
        persona=persona,
        persona_description=persona_desc,
    )

    raw      = call_llm(prompt, temperature=0.3, max_tokens=400)
    fallback = {
        "subject":    f"Re: {email['subject']}",
        "body":       "Thank you for your email. I will respond shortly.",
        "follow_ups": [],
    }
    result = parse_json_strict(raw, fallback=fallback, context=f"reply_{email_id}")

    if not validate_reply_output(result):
        result = fallback

    insert_draft(email_id, result["subject"], result["body"], persona)
    logger.info(f"generate_reply | {email_id} persona={persona}")

    return {
        "subject":    result.get("subject"),
        "body":       result.get("body"),
        "follow_ups": result.get("follow_ups", []),
        "persona":    persona,
    }