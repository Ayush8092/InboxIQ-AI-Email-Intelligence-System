"""
Categorized Emails Tab — fixed version.

Changes:
1. NO resume upload in this tab (moved to Job Task tab)
2. Full email content displayed correctly
3. Job emails always visible
4. Handles base64-encoded email bodies
5. All email categories show full content
"""
import streamlit as st
from collections import Counter
from memory.repository import get_all_emails, get_all_processed
from utils.helpers import priority_label, truncate
from utils.email_cleaner import clean_email_body, parse_email_html

_JOB_CATEGORIES = {
    "Job / Recruitment","Job","job","job_recruitment","Job/Recruitment",
}

_PRIORITY_COLORS = {
    1:"#FF4444",2:"#FF8C00",3:"#FFD700",
    4:"#4169E1",5:"#808080",6:"#A9A9A9",7:"#C0C0C0",
}


def _get_source() -> str:
    return "gmail" if st.session_state.get("authenticated") else "demo"


def _get_readable_body(email: dict) -> str:
    """
    Get readable text from email body.
    Handles base64-encoded Gmail payloads.
    """
    body = email.get("body","")
    if not body:
        return ""
    result = parse_email_html(body)
    return result.get("text","")


def _render_full_email(email: dict, key_suffix: str = ""):
    """
    Render FULL email content inside an expander.
    Shows text + all links.
    """
    body   = email.get("body","")
    parsed = parse_email_html(body) if body else {}
    text   = parsed.get("text","")
    links  = parsed.get("links",[])
    jlinks = parsed.get("job_links",[])

    col_view, col_toggle = st.columns([3,1])
    with col_toggle:
        show_raw = st.checkbox(
            "Raw",
            value=False,
            key=f"raw_{email.get('id','')[:15]}_{key_suffix}",
            help="Show raw email body",
        )

    with col_view:
        if show_raw:
            st.text_area(
                "Raw body",
                value=body[:3000],
                height=250,
                disabled=True,
                key=f"rawbody_{email.get('id','')[:15]}_{key_suffix}",
            )
        elif text:
            st.text_area(
                "Email Content",
                value=text,
                height=250,
                disabled=True,
                key=f"cleanbody_{email.get('id','')[:15]}_{key_suffix}",
            )
        else:
            # Last resort: show first 2000 chars of raw body
            if body:
                st.text_area(
                    "Email body (raw fallback)",
                    value=body[:2000],
                    height=200,
                    disabled=True,
                    key=f"fallback_{email.get('id','')[:15]}_{key_suffix}",
                )
            else:
                st.caption("_No email content available_")

    # Show job links
    if jlinks:
        st.markdown("**🔗 Job Apply Links:**")
        for lnk in jlinks[:5]:
            st.markdown(f"- [{lnk[:90]}]({lnk})")
    elif links:
        st.markdown("**🔗 Links in email:**")
        for lnk in links[:3]:
            st.markdown(f"- [{lnk[:90]}]({lnk})")


def _render_email_card(email: dict, processed: dict | None, idx: int):
    """Render one email card with expandable full content."""
    p       = processed or {}
    subject = email.get("subject","(No Subject)")
    sender  = email.get("sender","")
    cat     = p.get("category","Not processed")
    pr      = p.get("priority")
    task    = p.get("task","")
    summary = p.get("summary","")
    color   = _PRIORITY_COLORS.get(pr,"#808080")

    with st.container():
        c1, c2 = st.columns([5,1])
        with c1:
            st.markdown(f"**{truncate(subject, 80)}**")
            st.caption(f"From: {sender} &nbsp;|&nbsp; {cat}")
            if summary and summary not in ("—",""):
                st.info(f"💡 {summary}")
            if task and task not in ("—",""):
                st.markdown(f"📌 **Task:** {truncate(task, 100)}")
        with c2:
            if pr:
                st.markdown(
                    f'<div style="background:{color};color:white;'
                    f'border-radius:6px;padding:4px 8px;'
                    f'text-align:center;font-size:11px;">'
                    f'{priority_label(pr)}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📖 View Full Email", expanded=False):
            _render_full_email(email, key_suffix=str(idx))

        st.markdown("---")


def render_categorized_tab():
    st.header("📂 Categorized Emails")

    source      = _get_source()
    emails_list = get_all_emails(source=source)
    proc_list   = get_all_processed(source=source)
    proc_map    = {p["email_id"]: p for p in proc_list}

    if not emails_list:
        st.info(
            "No emails available. "
            + ("Click '📥 Load Gmail' in sidebar."
               if source == "gmail"
               else "Go to **📥 Inbox** tab and click **🚀 Process Emails**.")
        )
        return

    processed_count = sum(1 for e in emails_list if e["id"] in proc_map)
    if processed_count == 0:
        st.warning(
            "⚠️ Emails not processed yet. "
            "Go to **📥 Inbox** and click **🚀 Process Emails** first, "
            "then come back here to view by category."
        )
        # Still show all emails even if unprocessed
        st.subheader(f"All Emails ({len(emails_list)}) — unprocessed")
        for i, email in enumerate(emails_list):
            _render_email_card(email, None, i)
        return

    # ── Category counts ───────────────────────────────────────────────────────
    cat_counts: Counter = Counter()
    cat_counts["All"]   = len(emails_list)
    for email in emails_list:
        cat = proc_map.get(email["id"],{}).get("category")
        if cat:
            cat_counts[cat] += 1

    sorted_cats = ["All"] + sorted(
        [c for c in cat_counts if c != "All"],
        key=lambda x: -cat_counts[x],
    )
    labels = {c: f"{c} ({cat_counts.get(c,0)})" for c in sorted_cats}
    labels["All"] = f"All ({len(emails_list)})"

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3,2])
    with c1:
        sel_label = st.selectbox(
            "Filter by Category",
            list(labels.values()),
            index=0,
            key="cat_filter",
        )
        sel_cat = next((k for k, v in labels.items() if v == sel_label), "All")
    with c2:
        count = (
            len(emails_list)
            if sel_cat == "All"
            else cat_counts.get(sel_cat, 0)
        )
        st.metric("Showing", f"{count} emails")

    # ── Distribution ─────────────────────────────────────────────────────────
    with st.expander("📊 Category Distribution", expanded=False):
        total = max(processed_count, 1)
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            if cat == "All":
                continue
            pct = cnt / total * 100
            ca, cb, cc = st.columns([3,5,1])
            with ca: st.write(cat)
            with cb: st.progress(pct/100)
            with cc: st.write(str(cnt))

    st.divider()

    # ── Filter emails ─────────────────────────────────────────────────────────
    if sel_cat == "All":
        filtered = emails_list
    else:
        # Primary filter: by processed category
        filtered = [
            e for e in emails_list
            if proc_map.get(e["id"],{}).get("category") == sel_cat
        ]
        # If nothing matched but it's a job category, include all unmatched job emails
        if not filtered and sel_cat in _JOB_CATEGORIES:
            # Show all emails from known job senders as fallback
            job_senders = ["glassdoor","naukri","linkedin","indeed","monster","resume.io"]
            filtered = [
                e for e in emails_list
                if any(js in e.get("sender","").lower() for js in job_senders)
            ]

    if not filtered:
        st.info(f"No emails found in '{sel_cat}'.")
        return

    st.subheader(
        f"{'All Emails' if sel_cat == 'All' else sel_cat} ({len(filtered)})"
    )
    st.caption("Click '📖 View Full Email' on any email to see complete content")

    for i, email in enumerate(filtered):
        _render_email_card(email, proc_map.get(email["id"]), i)