"""
Categorized Emails Tab — Fixed.
Uses get_display_text() for email display (shows structured job listing).
Uses clean_email_body() is separate — handled in tools/categorize.py.
"""
import streamlit as st
from collections import Counter
from memory.repository import get_all_emails, get_all_processed
from utils.helpers import priority_label, truncate
from utils.email_cleaner import parse_email_html, get_display_text

_JOB_SENDERS = [
    "glassdoor", "naukri", "linkedin", "indeed", "monster",
    "resume.io", "jobs@", "career@", "hiring@", "recruiter",
]
_JOB_CATEGORIES = {
    "Job / Recruitment", "Job", "job", "job_recruitment", "Job/Recruitment",
}
_PRIORITY_COLORS = {
    1: "#FF4444", 2: "#FF8C00", 3: "#FFD700",
    4: "#4169E1", 5: "#808080", 6: "#A9A9A9", 7: "#C0C0C0",
}


def _get_source() -> str:
    return "gmail" if st.session_state.get("authenticated") else "demo"


def _render_email_expander(email: dict, uid: str):
    """Render full email content in expander — shows structured job listing."""
    body = email.get("body", "")
    if not body:
        st.caption("_No email content available_")
        return

    # Parse once
    parsed    = parse_email_html(body)
    text      = parsed.get("text", "")      # structured display text
    job_cards = parsed.get("job_cards", [])
    all_links = parsed.get("links", [])
    job_links = parsed.get("job_links", [])

    # Raw toggle
    show_raw = st.checkbox("Show raw HTML", value=False, key=f"raw_{uid}")

    if show_raw:
        st.text_area("Raw HTML", value=body[:4000], height=250, disabled=True, key=f"rawb_{uid}")
        return

    # Show structured job cards if we found them
    if job_cards:
        st.markdown(f"**📋 {len(job_cards)} Jobs in this email:**")
        for i, card in enumerate(job_cards):
            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{i+1}. {card['role']}**")
                    parts = []
                    if card.get("company", "Unknown") != "Unknown":
                        parts.append(f"🏢 {card['company']}")
                    if card.get("location", "Not specified") != "Not specified":
                        parts.append(f"📍 {card['location']}")
                    if card.get("salary", "Not specified") != "Not specified":
                        parts.append(f"💰 {card['salary']}")
                    if parts:
                        st.caption(" | ".join(parts))
                    if card.get("skills"):
                        skill_html = " ".join(
                            f'<span style="background:#e2e8f0;padding:1px 6px;'
                            f'border-radius:10px;font-size:11px;">{s}</span>'
                            for s in card["skills"][:8]
                        )
                        st.markdown(skill_html, unsafe_allow_html=True)
                with c2:
                    if card.get("link"):
                        st.markdown(
                            f'<a href="{card["link"]}" target="_blank">'
                            f'<button style="background:#4285F4;color:white;'
                            f'border:none;padding:5px 10px;border-radius:4px;'
                            f'cursor:pointer;font-size:12px;">Apply</button></a>',
                            unsafe_allow_html=True,
                        )

    elif text:
        # Plain text display
        st.text_area("Email Content", value=text, height=250, disabled=True, key=f"clean_{uid}")
    else:
        # Absolute fallback
        st.text_area("Email body", value=body[:3000], height=200, disabled=True, key=f"fb_{uid}")

    # Show apply links
    display_links = job_links or all_links[:3]
    if display_links:
        ltype = "**🔗 Job Apply Links:**" if job_links else "**🔗 Links:**"
        st.markdown(ltype)
        for lnk in display_links[:5]:
            st.markdown(f"- [{lnk[:90]}]({lnk})")


def _render_email_card(email: dict, processed: dict | None, idx: int):
    p       = processed or {}
    subject = email.get("subject", "(No Subject)")
    sender  = email.get("sender", "")
    cat     = p.get("category", "Not processed")
    pr      = p.get("priority")
    task    = p.get("task", "")
    summary = p.get("summary", "")
    color   = _PRIORITY_COLORS.get(pr, "#9ca3af")
    uid     = f"{email.get('id', 'x')[:12]}_{idx}"

    with st.container():
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"**{truncate(subject, 80)}**")
            st.caption(f"From: {sender} &nbsp;|&nbsp; {cat}")
            if summary and summary not in ("—", ""):
                st.info(f"💡 {summary}")
            if task and task not in ("—", ""):
                st.markdown(f"📌 {truncate(task, 100)}")
        with c2:
            if pr:
                st.markdown(
                    f'<div style="background:{color};color:white;border-radius:6px;'
                    f'padding:4px 8px;text-align:center;font-size:11px;">'
                    f'{priority_label(pr)}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📖 View Full Email", expanded=False):
            _render_email_expander(email, uid)

        st.markdown("---")


def render_categorized_tab():
    st.header("📂 Categorized Emails")
    st.caption("Click '📖 View Full Email' to see complete content including all job listings")

    source      = _get_source()
    emails_list = get_all_emails(source=source)
    proc_list   = get_all_processed(source=source)
    proc_map    = {p["email_id"]: p for p in proc_list}

    if not emails_list:
        st.info(
            "No emails available. "
            + ("Click '📥 Load Gmail' in the sidebar."
               if source == "gmail"
               else "Go to **📥 Inbox** and click **🚀 Process Emails**.")
        )
        return

    processed_count = sum(1 for e in emails_list if e["id"] in proc_map)

    # Category counts
    cat_counts: Counter = Counter()
    cat_counts["All"] = len(emails_list)
    for email in emails_list:
        cat = proc_map.get(email["id"], {}).get("category")
        if cat:
            cat_counts[cat] += 1

    sorted_cats = ["All"] + sorted(
        [c for c in cat_counts if c != "All"],
        key=lambda x: -cat_counts[x],
    )
    labels = {}
    for c in sorted_cats:
        cnt = len(emails_list) if c == "All" else cat_counts.get(c, 0)
        labels[c] = f"{c} ({cnt})"

    # Controls
    c1, c2 = st.columns([3, 2])
    with c1:
        sel_label = st.selectbox("Filter by Category", list(labels.values()), index=0, key="cat_filter")
        sel_cat   = next((k for k, v in labels.items() if v == sel_label), "All")
    with c2:
        cnt = len(emails_list) if sel_cat == "All" else cat_counts.get(sel_cat, 0)
        st.metric("Showing", f"{cnt} emails")

    if processed_count == 0:
        st.warning("⚠️ Go to **📥 Inbox** and click **🚀 Process Emails** first.")

    # Distribution
    with st.expander("📊 Category Distribution", expanded=False):
        total = max(processed_count, 1)
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            if cat == "All":
                continue
            pct = n / total * 100
            ca, cb, cc = st.columns([3, 5, 1])
            with ca: st.write(cat)
            with cb: st.progress(pct / 100)
            with cc: st.write(str(n))

    st.divider()

    # Filter
    if sel_cat == "All":
        filtered = emails_list
    else:
        filtered = [
            e for e in emails_list
            if proc_map.get(e["id"], {}).get("category") == sel_cat
        ]
        # Job category fallback: match by sender
        if not filtered and sel_cat in _JOB_CATEGORIES:
            filtered = [
                e for e in emails_list
                if any(js in e.get("sender", "").lower() for js in _JOB_SENDERS)
            ]
        # Last resort: case-insensitive match
        if not filtered:
            sel_norm = sel_cat.lower().replace(" / ", "/").replace(" ", "_")
            filtered = [
                e for e in emails_list
                if proc_map.get(e["id"], {}).get(
                    "category", ""
                ).lower().replace(" / ", "/").replace(" ", "_") == sel_norm
            ]

    if not filtered:
        st.info(f"No emails found in '{sel_cat}'.")
        return

    lbl = "All Emails" if sel_cat == "All" else sel_cat
    st.subheader(f"{lbl} ({len(filtered)})")

    for i, email in enumerate(filtered):
        _render_email_card(email, proc_map.get(email["id"]), i)