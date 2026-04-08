import pytest
from unittest.mock import patch, MagicMock
from tools.categorize import categorize_email
from tools.extract_tasks import extract_tasks
from tools.priority import compute_priority
from tools.summarize import summarize_email

MOCK_EMAIL = {
    "id": "test_001",
    "subject": "Q3 Budget deadline Friday",
    "body": "Please approve the Q3 budget report by Friday October 20, 2024. This is urgent.",
    "sender": "boss@company.com",
    "timestamp": "2024-10-14T09:00:00",
}

MOCK_PROCESSED = {
    "email_id": "test_001",
    "category": "Action Required",
    "priority": 2,
    "task": "Approve Q3 budget report",
    "deadline": "2024-10-20",
    "summary": None,
    "confidence": 0.9,
    "needs_review": 0,
}

@patch("tools.categorize.get_email", return_value=MOCK_EMAIL)
@patch("tools.categorize.get_prompt", return_value=None)
@patch("tools.categorize.call_llm", return_value='{"category": "Action Required", "confidence": 0.95}')
def test_categorize_email(mock_llm, mock_prompt, mock_get):
    result = categorize_email("test_001")
    assert result["category"] == "Action Required"
    assert result["confidence"] == 0.95

@patch("tools.extract_tasks.get_email", return_value=MOCK_EMAIL)
@patch("tools.extract_tasks.get_prompt", return_value=None)
@patch("tools.extract_tasks.call_llm", return_value='{"task": "Approve Q3 budget report", "deadline": "2024-10-20", "confidence": 0.9}')
def test_extract_tasks(mock_llm, mock_prompt, mock_get):
    result = extract_tasks("test_001")
    assert result["task"] == "Approve Q3 budget report"
    assert result["deadline"] == "2024-10-20"

@patch("tools.priority.get_email", return_value=MOCK_EMAIL)
@patch("tools.priority.get_processed", return_value=MOCK_PROCESSED)
def test_compute_priority_high(mock_proc, mock_get):
    result = compute_priority("test_001")
    assert result["priority"] <= 2  # boss + urgent keyword → high priority

@patch("tools.summarize.get_email", return_value=MOCK_EMAIL)
@patch("tools.summarize.get_prompt", return_value=None)
@patch("tools.summarize.call_llm", return_value="Boss requests Q3 budget approval by Friday.")
def test_summarize_email(mock_llm, mock_prompt, mock_get):
    result = summarize_email("test_001")
    assert "budget" in result["summary"].lower()