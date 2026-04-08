"""
Email Summarization Dashboard.
Aggregates emails into category groups, generates LLM summaries,
highlights key insights and urgent items.
"""
import streamlit as st
from collections import defaultdict
from datetime import date, datetime
from memory.repository import (
    get_all_emails, get_all_processed, get_all_feedback,
)
from utils.llm_client import call_llm
from utils.helpers import priority_label, confidence_label, truncate
from utils.cache import get_cached, set_cached
from config.constants import CATEGORIES, PRIORITY_LABELS


def _generate_group_summary(emails_in_group: list[dict], category: str) -> str:
    """Generate LLM summary for a group of emails in the same category."""
    if not emails_in_group:
        return ""

    snippets = []
    for e in emails_in_group[:5]:
        snippets.append(f"- {e['subject']} (from {e['sender']})")

    prompt = (
        f"You are summarizing a group of {len(emails_in_group)} emails "
        f"categorized as '{category}'.\n\n"
        f"Email subjects:\n" + "\n".join(snippets) + "\n\n"
        f"Write a 1-2 sentence executive summary of what this group of emails "
        f"is about and what actions if any are needed. Be concise and direct."
    )

    cache_key = f"dash_summary_{category}_{len(emails_in_group)}"
    cached    = get_cached(prompt)
    if cached:
        return cached

    result = call_llm(prompt, temperature=0.1, max_tokens=100, use_cache=True)
    return result or f"{len(emails_in_group)} emails in {category}."


def _get_key_insights(all_processed: list[dict], emails_map: dict) -> list[dict]:
    """
    Extract key insights across all emails:
    - Overdue tasks
    - Critical priority emails without tasks
    - High-confidence categorizations
    - Recent corrections
    """
    insights = []
    today    = date.today()

    for p in all_processed:
        email_id = p.get("email_id")
        email    = emails_map.get(email_id, {})
        subject  = email.get("subject","")
        priority = p.get("priority", 5)
        deadline = p.get("deadline")
        task     = p.get("task")
        category = p.get("category","")

        # Overdue tasks
        if deadline and task:
            try:
                dl   = datetime.strptime(deadline, "%Y-%m-%d").date()
                days = (today - dl).days
                if days > 0:
                    insights.append({
                        "type":    "overdue",
                        "icon":    "🔴",
                        "label":   "Overdue Task",
                        "message": f"{truncate(task,50)} — was due {days} day{'s' if days>1 else ''} ago",
                        "email_id": email_id,
                        "subject": subject,
                        "priority": 1,
                    })
            except Exception:
                pass

        # Critical emails without tasks
        if priority == 1 and not task and category not in {"Newsletter","Social / Notification"}:
            insights.append({
                "type":    "missing_task",
                "icon":    "⚠️",
                "label":   "Critical — No Task",
                "message": f"{truncate(subject,50)} — critical email with no extracted task",
                "email_id": email_id,
                "subject": subject,
                "priority": 2,
            })

        # Upcoming deadlines (within 2 days)
        if deadline and task:
            try:
                dl   = datetime.strptime(deadline, "%Y-%m-%d").date()
                days = (dl - today).days
                if 0 <= days <= 2:
                    insights.append({
                        "type":    "urgent_deadline",
                        "icon":    "⏰",
                        "label":   f"Due {'Today' if days==0 else 'Tomorrow' if days==1 else 'In 2 Days'}",
                        "message": f"{truncate(task,50)}",
                        "email_id": email_id,
                        "subject": subject,
                        "priority": 3,
                    })
            except Exception:
                pass

    # Sort by priority
    insights.sort(key=lambda x: x["priority"])
    return insights[:10]  # top 10


def render_dashboard_tab():
    st.header("📊 Email Summary Dashboard")

    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    emails_list = get_all_emails()
    processed   = get_all_processed()
    emails_map  = {e["id"]: e for e in emails_list}
    proc_map    = {p["email_id"]: p for p in processed}

    if not processed:
        st.info("No emails processed yet. Go to Inbox tab and click 'Process All Emails'.")
        return

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    total     = len(processed)
    critical  = sum(1 for p in processed if p.get("priority") == 1)
    high      = sum(1 for p in processed if p.get("priority") == 2)
    with_task = sum(1 for p in processed if p.get("task"))
    overdue   = 0
    today     = date.today()

    for p in processed:
        dl = p.get("deadline")
        if dl:
            try:
                if datetime.strptime(dl, "%Y-%m-%d").date() < today:
                    overdue += 1
            except Exception:
                pass

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📧 Total",         total)
    c2.metric("🔴 Critical",      critical, delta=f"{critical/total*100:.0f}%" if total else "0%")
    c3.metric("🟠 High Priority", high)
    c4.metric("✅ With Tasks",    with_task)
    c5.metric("⏰ Overdue",       overdue)

    st.divider()

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.subheader("🔍 Key Insights")
    insights = _get_key_insights(processed, emails_map)

    if insights:
        for ins in insights:
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown(f"### {ins['icon']}")
            with col2:
                st.markdown(f"**{ins['label']}:** {ins['message']}")
                st.caption(f"Email: {ins['email_id']} | {truncate(ins['subject'],60)}")
    else:
        st.success("✅ No critical insights — inbox looks healthy!")

    st.divider()

    # ── Category summaries ────────────────────────────────────────────────────
    st.subheader("📂 Category Summary")

    groups = defaultdict(list)
    for p in processed:
        cat   = p.get("category","Unknown")
        email = emails_map.get(p["email_id"],{})
        groups[cat].append({**p, **{"subject": email.get("subject",""), "sender": email.get("sender","")}})

    for cat in CATEGORIES:
        group = groups.get(cat, [])
        if not group:
            continue

        critical_in_group = sum(1 for g in group if g.get("priority",7) <= 2)
        has_overdue       = any(
            g.get("deadline") and
            datetime.strptime(g["deadline"], "%Y-%m-%d").date() < today
            for g in group
            if g.get("deadline")
        )

        header_extra = ""
        if critical_in_group:
            header_extra += f" 🔴 {critical_in_group} critical"
        if has_overdue:
            header_extra += " ⏰ overdue"

        with st.expander(f"**{cat}** — {len(group)} emails{header_extra}"):
            # LLM group summary
            summary = _generate_group_summary(group, cat)
            if summary:
                st.info(f"💡 {summary}")

            # Email list in this group
            for g in sorted(group, key=lambda x: x.get("priority",7)):
                p_label = priority_label(g.get("priority"))
                c_label = confidence_label(g.get("confidence"))
                task    = g.get("task","")
                dl      = g.get("deadline","")

                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{truncate(g.get('subject',''),45)}**")
                    st.caption(g.get("sender",""))
                with col2:
                    st.write(p_label)
                    st.caption(c_label)
                with col3:
                    if task:
                        st.write(f"📌 {truncate(task,35)}")
                    if dl:
                        st.caption(f"📅 {dl}")

                st.divider()

    # ── Sender analysis ───────────────────────────────────────────────────────
    st.subheader("👤 Top Senders")
    sender_stats = defaultdict(lambda: {"count":0,"critical":0,"tasks":0})
    for p in processed:
        email  = emails_map.get(p["email_id"],{})
        sender = email.get("sender","unknown")
        sender_stats[sender]["count"]    += 1
        if p.get("priority",7) <= 2:
            sender_stats[sender]["critical"] += 1
        if p.get("task"):
            sender_stats[sender]["tasks"]    += 1

    sorted_senders = sorted(
        sender_stats.items(), key=lambda x: (-x[1]["critical"], -x[1]["count"])
    )[:8]

    if sorted_senders:
        rows = []
        for sender, stats in sorted_senders:
            rows.append({
                "Sender":   sender,
                "Emails":   stats["count"],
                "Critical": stats["critical"],
                "Tasks":    stats["tasks"],
            })
        st.dataframe(rows, use_container_width=True)