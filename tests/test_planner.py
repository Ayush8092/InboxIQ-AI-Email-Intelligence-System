import pytest
from unittest.mock import patch
from agent.planner import run_planner

MOCK_EMAIL = {
    "id": "test_001",
    "subject": "Urgent: server down",
    "body": "Production is down immediately fix it.",
    "sender": "cto@company.com",
    "timestamp": "2024-10-14T13:00:00",
}

VALID_PLAN_JSON = '''{
  "tools_to_call": ["categorize_email", "compute_priority", "summarize_email"],
  "skip_reasons": {"extract_tasks": "Alert email", "generate_reply": "Alert category"},
  "needs_review": false,
  "explanation": "This is an alert email. Categorize and prioritize only."
}'''

@patch("agent.planner.get_processed", return_value=None)
@patch("agent.planner.get_prompt", return_value=None)
@patch("agent.planner.call_llm", return_value=VALID_PLAN_JSON)
def test_planner_returns_valid_plan(mock_llm, mock_prompt, mock_proc):
    plan = run_planner(MOCK_EMAIL, "Handle this email")
    assert "tools_to_call" in plan
    assert isinstance(plan["tools_to_call"], list)
    assert "explanation" in plan

@patch("agent.planner.get_processed", return_value=None)
@patch("agent.planner.get_prompt", return_value=None)
@patch("agent.planner.call_llm", return_value="not valid json at all %%%")
def test_planner_fallback_on_bad_json(mock_llm, mock_prompt, mock_proc):
    plan = run_planner(MOCK_EMAIL, "Handle this email")
    assert "tools_to_call" in plan
    assert plan["needs_review"] is True  # fallback sets needs_review=True