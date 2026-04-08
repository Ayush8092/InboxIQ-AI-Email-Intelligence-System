import pytest
import os
import tempfile
from memory.db import init_db, get_connection
from memory import repository as repo

# Use a temp DB for tests
os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")

SAMPLE_EMAIL = {
    "id": "t001",
    "subject": "Test subject",
    "body": "Test body",
    "sender": "test@test.com",
    "timestamp": "2024-10-14T10:00:00",
}

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield

def test_insert_and_get_email():
    repo.insert_email(SAMPLE_EMAIL)
    email = repo.get_email("t001")
    assert email is not None
    assert email["subject"] == "Test subject"

def test_upsert_processed():
    repo.insert_email(SAMPLE_EMAIL)
    repo.upsert_processed({
        "email_id": "t001", "category": "Action Required",
        "priority": 2, "task": "Do something", "deadline": "2024-10-20",
        "summary": "A test.", "confidence": 0.9, "needs_review": 0,
    })
    p = repo.get_processed("t001")
    assert p["category"] == "Action Required"
    assert p["priority"] == 2

def test_insert_draft():
    repo.insert_email(SAMPLE_EMAIL)
    repo.insert_draft("t001", "Re: Test", "Thanks!", "Formal")
    drafts = repo.get_drafts("t001")
    assert len(drafts) >= 1
    assert drafts[0]["persona"] == "Formal"

def test_feedback():
    repo.insert_email(SAMPLE_EMAIL)
    repo.insert_feedback("t001", "category", "General Info", "Action Required")
    fb = repo.get_all_feedback()
    assert any(f["email_id"] == "t001" for f in fb)