import streamlit as st
import csv
import io
from memory.repository import (
    get_all_emails, get_all_processed,
    update_processed_field, insert_feedback,
    get_feedback_preferences, get_email, get_processed,
    get_drafts,
)
from utils.helpers import priority_label, confidence_label, truncate
from agent.orchestrator import orchestrator
from config.constants import PERSONAS

_TYPE_ICONS = {
    "task":           "✅",
    "multi_step":     "📋",
    "reminder":       "🔔",
    "calendar_event": "📅",
    "informational":  "ℹ️",
}


def _get_active_source() -> str | None:
    """
    Returns the email source to display based on auth state.
    Authenticated → 'gmail'
    Demo mode     → 'demo'
    """
    return "gmail" if st.session_state.get("authenticated") else "demo"


def render_inbox_tab():
    st.header("Email Inbox")

    # ── Source indicator ──────────────────────────────────────────────────────
    source = _get_active_source()
    if source == "gmail":
        st.success("📨 Showing your **Gmail emails** (real inbox)")
    else:
        st.info("🎭 Showing **Demo emails** (sample data)")

    col1, col2 = st.columns([2, 2])
    with col1:
        persona = st.selectbox("Persona", PERSONAS, key="inbox_persona")
    with col2:
        dry_run = st.checkbox("Dry Run Mode", key="inbox_dry")

    if st.button("🚀 Process Emails", type="primary"):
        with st.spinner("Processing emails..."):
            try:
                # Only process emails from active source
                emails = get_all_emails(source=source)
                if not emails:
                    if source == "gmail":
                        st.warning(
                            "No Gmail emails loaded. "
                            "Click '📥 Load Gmail' in the sidebar first."
                        )
                    else:
                        st.warning("No demo emails found.")
                else:
                    results = orchestrator.handle_all_emails(
                        emails, persona=persona, dry_run=dry_run
                    )
                    ok  = sum(1 for r in results if "error" not in r)
                    err = sum(1 for r in results if "error" in r)
                    nr  = sum(
                        1 for r in results
                        if "error" not in r
                        and r.get("final_state",{}).get("needs_review")
                    )
                    st.success(
                        f"✅ Processed {ok} | ⚠️ {nr} flagged | ❌ {err} errors"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Inbox table ───────────────────────────────────────────────────────────
    st.subheader("Inbox Table")

    emails_list = get_all_emails(source=source)
    processed   = {p["email_id"]: p for p in get_all_processed(source=source)}
    emails_map  = {e["id"]: e for e in emails_list}

    rows = []
    for email in emails_list:
        eid    = email["id"]
        p      = processed.get(eid, {})
        nr     = p.get("needs_review")
        t_type = p.get("task_type","")
        icon   = _TYPE_ICONS.get(t_type,"")
        steps  = p.get("steps") or []

        rows.append({
            "ID":         eid,
            "Subject":    truncate(email["subject"], 40),
            "Sender":     email["sender"],
            "Category":   p.get("category") or "—",
            "Priority":   priority_label(p.get("priority")),
            "Confidence": confidence_label(p.get("confidence")),
            "Type":       f"{icon} {t_type}" if t_type else "—",
            "Task":       truncate(p.get("task") or "—", 35),
            "Steps":      str(len(steps)) if steps else "—",
            "Deadline":   p.get("deadline") or "—",
            "Review":     "⚠️" if nr else "✅",
        })

    if rows:
        st.dataframe(rows, use_container_width=True)

        # ── CSV Export (source-filtered) ──────────────────────────────────
        csv_buffer = io.StringIO()
        writer     = csv.DictWriter(csv_buffer, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

        source_label = "gmail" if source == "gmail" else "demo"
        st.download_button(
            label=f"⬇️ Export {source_label} emails as CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name=f"aeoa_{source_label}_emails.csv",
            mime="text/csv",
        )
    else:
        if source == "gmail":
            st.info(
                "No Gmail emails loaded yet. "
                "Click '📥 Load Gmail' in the sidebar."
            )
        else:
            st.info("No demo emails found.")

    st.divider()

    # ── Process single email ──────────────────────────────────────────────────
    st.subheader("Process Single Email")

    all_ids = [e["id"] for e in emails_list]
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if all_ids:
            email_id = st.selectbox(
                "Select Email", all_ids,
                format_func=lambda x: f"{x} — {emails_map.get(x,{}).get('subject','')[:50]}",
                key="single_id_select",
            )
        else:
            email_id = st.text_input(
                "Email ID", placeholder="email_001", key="single_id"
            )
    with col2:
        single_persona = st.selectbox("Persona", PERSONAS, key="single_persona")
    with col3:
        single_dry = st.checkbox("Dry Run", key="single_dry")

    if st.button("▶ Process Email", type="primary"):
        if not email_id:
            st.warning("Please select or enter an email ID.")
        else:
            raw_email = get_email(email_id)
            if raw_email:
                with st.expander("📧 Original Email", expanded=True):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.write(f"**From:** {raw_email['sender']}")
                        st.write(f"**Subject:** {raw_email['subject']}")
                        st.write(f"**Date:** {raw_email.get('timestamp','')[:16]}")
                        src_badge = "🔵 Gmail" if raw_email.get("source") == "gmail" else "🟡 Demo"
                        st.write(f"**Source:** {src_badge}")
                    with col_b:
                        st.text_area(
                            "Body",
                            value=raw_email["body"],
                            height=120,
                            disabled=True,
                            key="email_body_view",
                        )

            with st.spinner("Processing..."):
                result = orchestrator.handle_email(
                    email_id,
                    persona=single_persona,
                    dry_run=single_dry,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                plan   = result["plan"]
                state  = result["final_state"]
                nr     = state.get("needs_review")
                rr     = state.get("review_reason") or ""
                t_type = state.get("task_type","task")
                steps  = state.get("steps") or []
                icon   = _TYPE_ICONS.get(t_type,"✅")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📊 Result")
                    st.write(f"**Category:** {state.get('category','—')}")
                    st.write(f"**Priority:** {priority_label(state.get('priority'))}")
                    st.write(f"**Confidence:** {confidence_label(state.get('confidence'))}")
                    st.write(f"**Summary:** {state.get('summary') or '—'}")
                    st.markdown(f"#### {icon} Task ({t_type})")
                    st.info(state.get("task") or "—")
                    if steps:
                        st.markdown("**Steps:**")
                        for i, s in enumerate(steps, 1):
                            st.write(f"{i}. {s}")
                    if state.get("deadline"):
                        st.write(f"📅 **Deadline:** `{state['deadline']}`")
                    if nr:
                        st.warning("⚠️ **Needs Review**")
                        for r in rr.split(";"):
                            if r.strip():
                                st.write(f"  - {r.strip()}")
                    else:
                        st.success("✅ Auto-handled")

                with col2:
                    st.markdown("#### 🧠 Planner Trace")
                    st.write(f"**Explanation:** {plan.get('explanation','—')}")
                    tools = plan.get("tools_to_call",[])
                    st.write(f"**Tools:** {', '.join(tools) if tools else 'None'}")
                    skips = plan.get("skip_reasons",{})
                    if skips:
                        st.write("**Skipped:**")
                        for t, r in skips.items():
                            st.write(f"- `{t}`: {r}")

                drafts = get_drafts(email_id)
                if drafts:
                    with st.expander("✉️ Reply Draft"):
                        d = drafts[0]
                        st.write(f"**Subject:** {d['subject']}")
                        st.text_area("Body", value=d["body"], height=150, key="draft_view")

    st.divider()

    # ── Feedback ──────────────────────────────────────────────────────────────
    st.subheader("✏️ Correct a Field (Feedback Learning)")

    col1, col2, col3 = st.columns(3)
    with col1:
        fb_email = st.text_input("Email ID", key="fb_email")
    with col2:
        fb_field = st.selectbox(
            "Field", ["category","priority","task","deadline"], key="fb_field"
        )
    with col3:
        fb_value = st.text_input("Correct Value", key="fb_value")

    if st.button("💾 Save Correction"):
        if not fb_email.strip() or not fb_value.strip():
            st.warning("Fill all fields.")
        else:
            p         = get_processed(fb_email.strip()) or {}
            old_value = str(p.get(fb_field,""))
            update_processed_field(fb_email.strip(), fb_field, fb_value.strip())
            insert_feedback(fb_email.strip(), fb_field, old_value, fb_value.strip())
            st.success(f"✅ Saved: `{fb_field}` → `{fb_value}`")

    st.divider()

    # ── Needs review ──────────────────────────────────────────────────────────
    st.subheader("⚠️ Needs Review")
    all_p   = get_all_processed(source=source)
    nr_list = [p for p in all_p if p.get("needs_review")]
    if nr_list:
        for p in nr_list:
            t_type = p.get("task_type","task")
            icon   = _TYPE_ICONS.get(t_type,"")
            with st.expander(
                f"📧 {p['email_id']} — {p.get('category') or 'Uncategorized'} "
                f"| {confidence_label(p.get('confidence'))}"
            ):
                st.write(f"**{icon} Task:** {p.get('task') or '—'}")
                rr = p.get("review_reason","")
                if rr:
                    st.warning("**Reasons:**")
                    for r in rr.split(";"):
                        if r.strip():
                            st.write(f"- {r.strip()}")
    else:
        st.success("✅ No emails need review.")