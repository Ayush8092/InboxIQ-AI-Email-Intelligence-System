import json
from datetime import datetime

def utcnow_iso() -> str:
    return datetime.utcnow().isoformat()

def priority_label(priority: int) -> str:
    labels = {
        1: "🔴 Critical", 2: "🟠 High",    3: "🟡 Medium",
        4: "🔵 Low",      5: "⚪ Very Low", 6: "⚪ Minimal", 7: "⚪ Negligible",
    }
    return labels.get(priority, "⚪ Unknown")

def confidence_label(confidence) -> str:
    if confidence is None:
        return "—"
    c = float(confidence)
    if c >= 0.85: return f"✅ {c:.0%}"
    if c >= 0.50: return f"⚠️ {c:.0%}"
    return f"❌ {c:.0%}"

def truncate(text: str, max_len: int = 80) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len - 3] + "..."

def load_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)