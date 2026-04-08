"""
Categorized Emails Tab.

Fixes:
- Clean body display (HTML toggle)
- Correct job detection
- Apply link display from anchor tags
- Skill validation in job cards
- Fixed 100% match score display
- Proper empty state messages
"""
import streamlit as st
from collections import Counter
from memory.repository import get_all_emails, get_all_processed
from utils.helpers import priority_label, truncate
from utils.email_cleaner import clean_email_body, parse_email_html

_JOB_CATEGORIES = {
    "Job / Recruitment", "Job", "job", "job_recruitment", "Job/Recruitment",
}

_PRIORITY_COLORS = {
    1: "#FF4444", 2: "#FF8C00", 3: "#FFD700",
    4: "#4169E1", 5: "#808080", 6: "#A9A9A9", 7: "#C0C0C0",
}

_REC_COLORS = {
    "Strong Match":  "#22c55e",
    "Good Match":    "#84cc16",
    "Partial Match": "#f59e0b",
    "Weak Match":    "#ef4444",
}


def _get_source() -> str:
    return "gmail" if st.session_state.get("authenticated") else "demo"


def _render_job_card(job: dict, idx: int, show_match: bool = False):
    """Render job card with apply links and validated skills."""
    match     = job.get("match", {})
    score     = match.get("match_score", 0) if show_match else None
    rec       = match.get("recommendation", "") if show_match else ""
    rec_color = _REC_COLORS.get(rec, "#808080")
    ready     = match.get("ready_to_apply", False) if show_match else False

    with st.container():
        col_t, col_s = st.columns([4, 1])

        with col_t:
            st.markdown(f"### 💼 {job.get('role', 'Unknown Role')}")
            parts = []
            if job.get("company") and job["company"] != "Unknown":
                parts.append(f"🏢 **{job['company']}**")
            if job.get("location") and job["location"] != "Not specified":
                parts.append(f"📍 {job['location']}")
            if job.get("salary") and job["salary"] != "Not specified":
                parts.append(f"💰 {job['salary']}")
            if parts:
                st.markdown(" &nbsp;|&nbsp; ".join(parts))

        with col_s:
            if show_match and score is not None:
                st.markdown(
                    f"""<div style="background:{rec_color};color:white;
                    border-radius:8px;padding:8px;text-align:center;
                    font-weight:bold;font-size:22px;">{score}%</div>
                    <div style="text-align:center;font-size:11px;
                    color:{rec_color};font-weight:bold;margin-top:4px">{rec}</div>""",
                    unsafe_allow_html=True,
                )
                if ready:
                    st.success("✅ Ready")

        # Skills — only show validated ones
        skills = job.get("skills", [])
        if skills:
            tags = " ".join(
                f'<span style="background:#e2e8f0;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;">{s}</span>'
                for s in skills[:12]
            )
            st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)
        else:
            st.caption("_No skills extracted_")

        # Description
        if job.get("description"):
            st.caption(job["description"])

        # Match breakdown
        if show_match and match:
            c1, c2 = st.columns(2)
            with c1:
                if match.get("matched_skills"):
                    st.markdown(
                        "✅ **Matched:** " +
                        ", ".join(f"`{s}`" for s in match["matched_skills"][:6])
                    )
                if match.get("fit_reason"):
                    st.caption(f"💡 {match['fit_reason']}")
            with c2:
                if match.get("missing_skills"):
                    st.markdown(
                        "⚠️ **Missing:** " +
                        ", ".join(f"`{s}`" for s in match["missing_skills"][:6])
                    )
                if match.get("gaps"):
                    st.caption(f"📌 {match['gaps']}")

            suggestions = match.get("skill_gap_suggestions", [])
            if suggestions:
                with st.expander("📚 Skill Gap Learning Path"):
                    for sg in suggestions[:3]:
                        skill    = sg.get("skill", "")
                        resource = sg.get("resource", "")
                        if skill:
                            st.write(f"• **{skill}** — {resource}")

        # Apply buttons
        all_links = job.get("all_links") or ([job["link"]] if job.get("link") else [])
        if all_links:
            st.markdown("**Apply:**")
            cols = st.columns(min(len(all_links), 3))
            for i, link in enumerate(all_links[:3]):
                with cols[i]:
                    label = f"🔗 Link {i+1}" if i > 0 else "🔗 Apply Now"
                    st.markdown(
                        f'<a href="{link}" target="_blank">'
                        f'<button style="background:#4285F4;color:white;border:none;'
                        f'padding:6px 14px;border-radius:4px;cursor:pointer;'
                        f'font-size:13px;width:100%;">{label}</button></a>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("_No apply link found_")

        # Source badge
        src = job.get("source", "")
        if src == "rule":
            st.caption("📋 Extracted by rules")
        elif src == "llm":
            st.caption("🤖 Extracted by AI")
        if job.get("scraped"):
            st.caption("🌐 Enriched from web")

        st.markdown("---")


def _render_email_card(email: dict, processed: dict | None, show_raw: bool):
    """Render a standard email card with clean body."""
    p       = processed or {}
    subject = email.get("subject", "")
    sender  = email.get("sender", "")
    cat     = p.get("category", "—")
    pr      = p.get("priority")
    task    = p.get("task", "")
    summary = p.get("summary", "")
    color   = _PRIORITY_COLORS.get(pr, "#808080")

    body_raw = email.get("body", "")
    if show_raw:
        body_display = body_raw[:500]
    else:
        body_display = clean_email_body(body_raw, max_chars=200)

    with st.container():
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"**{truncate(subject, 70)}**")
            st.caption(f"From: {sender} &nbsp;|&nbsp; {cat}")
            if summary and summary not in ("—", ""):
                st.info(f"💡 {summary}")
            elif body_display:
                st.caption(truncate(body_display, 150))
            if task and task not in ("—", ""):
                st.markdown(f"📌 {truncate(task, 80)}")
        with c2:
            if pr:
                st.markdown(
                    f'<div style="background:{color};color:white;'
                    f'border-radius:6px;padding:4px 8px;text-align:center;'
                    f'font-size:12px;">{priority_label(pr)}</div>',
                    unsafe_allow_html=True,
                )
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
            + ("Click '📥 Load Gmail' in sidebar." if source == "gmail"
               else "Go to **📥 Inbox** tab and click **🚀 Process Emails**.")
        )
        return

    processed_count = sum(1 for e in emails_list if e["id"] in proc_map)
    if processed_count == 0:
        st.warning(
            "⚠️ No emails processed yet. "
            "Go to **📥 Inbox** tab and click **🚀 Process Emails** first."
        )
        return

    # ── Build category counts ─────────────────────────────────────────────────
    cat_counts: Counter = Counter()
    cat_counts["All"]   = processed_count

    for email in emails_list:
        cat = proc_map.get(email["id"], {}).get("category")
        if cat:
            cat_counts[cat] += 1

    sorted_cats  = ["All"] + sorted(
        [c for c in cat_counts if c != "All"],
        key=lambda x: -cat_counts[x],
    )
    labels = {c: f"{c} ({cat_counts[c]})" for c in sorted_cats}

    # ── Controls row ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        sel_label = st.selectbox(
            "Filter by Category",
            list(labels.values()),
            index=0,
            key="cat_filter",
        )
        sel_cat = next((k for k, v in labels.items() if v == sel_label), "All")
    with c2:
        show_raw = st.checkbox(
            "Show raw HTML",
            value=False,
            key="show_raw",
            help="Toggle between cleaned text and raw HTML",
        )
    with c3:
        count = cat_counts.get(sel_cat, 0) if sel_cat != "All" else processed_count
        st.metric("Emails shown", count)

    # Filter
    if sel_cat == "All":
        filtered = [e for e in emails_list if e["id"] in proc_map]
    else:
        filtered = [
            e for e in emails_list
            if proc_map.get(e["id"], {}).get("category") == sel_cat
        ]

    # ── Distribution bar ──────────────────────────────────────────────────────
    with st.expander("📊 Category Distribution", expanded=False):
        total = max(processed_count, 1)
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            if cat == "All":
                continue
            pct = count / total * 100
            ca, cb, cc = st.columns([3, 5, 1])
            with ca: st.write(cat)
            with cb: st.progress(pct / 100)
            with cc: st.write(str(count))

    st.divider()

    # ── Job intelligence ──────────────────────────────────────────────────────
    if sel_cat in _JOB_CATEGORIES and filtered:
        _render_job_intelligence(filtered, proc_map)
        return

    # ── Standard email view ───────────────────────────────────────────────────
    label = "All Emails" if sel_cat == "All" else sel_cat
    st.subheader(f"{label} ({len(filtered)})")

    if not filtered:
        st.info(f"No emails found in '{sel_cat}'.")
        return

    for email in filtered:
        _render_email_card(email, proc_map.get(email["id"]), show_raw)


def _render_job_intelligence(job_emails: list[dict], proc_map: dict):
    """Full job intelligence panel."""
    st.subheader(f"💼 Job Intelligence ({len(job_emails)} emails)")

    # ── Resume upload ─────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Resume")
    cu, cs = st.columns([2, 2])
    with cu:
        uploaded = st.file_uploader(
            "PDF or TXT",
            type=["pdf", "txt"],
            key="resume_uploader",
        )

    resume_data = st.session_state.get("parsed_resume")

    if uploaded:
        if st.session_state.get("resume_filename") != uploaded.name or not resume_data:
            with st.spinner("Parsing resume..."):
                from services.job_service import parse_resume
                fb          = uploaded.read()
                ft          = "pdf" if uploaded.name.endswith(".pdf") else "txt"
                resume_data = parse_resume(fb, ft)
                st.session_state["parsed_resume"]   = resume_data
                st.session_state["resume_filename"] = uploaded.name

    with cs:
        if resume_data and resume_data.get("skills"):
            name = resume_data.get("name") or "Your Resume"
            st.success(f"✅ **{name}**")
            st.caption(
                f"Skills: {len(resume_data['skills'])} | "
                f"Exp: {resume_data.get('experience_years', 0)} yrs | "
                f"Role: {resume_data.get('current_role', '—')}"
            )
            with st.expander("📋 Parsed Resume"):
                tags = " ".join(
                    f'<span style="background:#dcfce7;padding:2px 8px;'
                    f'border-radius:10px;font-size:12px;margin:2px;">{s}</span>'
                    for s in resume_data["skills"][:20]
                )
                st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)
                if resume_data.get("summary"):
                    st.caption(resume_data["summary"])
                if resume_data.get("certifications"):
                    st.write("**Certs:** " + ", ".join(resume_data["certifications"][:5]))
        else:
            st.info("📤 Upload resume for match scores")

    st.divider()

    # ── Extraction controls ───────────────────────────────────────────────────
    st.markdown("### 🔍 Extract Jobs")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        enrich = st.checkbox(
            "🌐 Web enrichment",
            value=False,
            key="job_enrich",
            help="Fetch job pages (skips LinkedIn/Glassdoor)",
        )
    with cc2:
        max_e = st.slider(
            "Max emails",
            1, min(len(job_emails), 50),
            min(len(job_emails), 10),
            key="job_max",
        )
    with cc3:
        st.write("")
        extract_btn = st.button(
            "🚀 Extract Jobs",
            type="primary",
            use_container_width=True,
        )

    if extract_btn:
        from services.job_service import extract_jobs_from_email, enrich_job_with_scraping
        to_process = job_emails[:max_e]
        all_jobs   = []
        prog       = st.progress(0)
        status     = st.empty()

        for i, email in enumerate(to_process):
            status.text(f"Processing {i+1}/{len(to_process)}: {truncate(email.get('subject',''), 50)}")
            jobs = extract_jobs_from_email(email)
            if enrich:
                jobs = [enrich_job_with_scraping(j) for j in jobs]
            all_jobs.extend(jobs)
            prog.progress((i + 1) / len(to_process))

        # Deduplicate (role, company, location)
        seen, unique = set(), []
        for job in all_jobs:
            import re
            key = (
                re.sub(r'\s+', ' ', job.get("role", "")).lower()[:60],
                re.sub(r'\s+', ' ', job.get("company", "")).lower()[:40],
                job.get("location", "").lower()[:30],
            )
            if key not in seen and key[0]:
                seen.add(key)
                unique.append(job)

        st.session_state["extracted_jobs"] = unique
        st.session_state.pop("scored_jobs", None)
        prog.progress(1.0)
        status.success(f"✅ Found {len(unique)} unique jobs!")

    # ── Score ─────────────────────────────────────────────────────────────────
    extracted = st.session_state.get("extracted_jobs", [])
    if extracted and resume_data and resume_data.get("skills"):
        if st.button("🎯 Score Against Resume", type="secondary"):
            with st.spinner(f"Scoring {len(extracted)} jobs..."):
                from services.job_service import score_all_jobs
                scored = score_all_jobs(extracted, resume_data)
                st.session_state["scored_jobs"] = scored
            st.success(f"✅ Scored {len(scored)} jobs!")

    # ── Display ───────────────────────────────────────────────────────────────
    jobs_to_show = st.session_state.get("scored_jobs") or extracted
    show_match   = bool(
        st.session_state.get("scored_jobs") and
        resume_data and resume_data.get("skills")
    )

    if jobs_to_show:
        st.divider()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Jobs Found", len(jobs_to_show))
        with m2:
            if show_match:
                strong = sum(
                    1 for j in jobs_to_show
                    if j.get("match", {}).get("match_score", 0) >= 70
                )
                st.metric("Strong/Good Matches", strong)
        with m3:
            min_score = (
                st.slider("Min score", 0, 100, 0, key="min_s")
                if show_match else 0
            )

        if show_match and min_score > 0:
            jobs_to_show = [
                j for j in jobs_to_show
                if j.get("match", {}).get("match_score", 0) >= min_score
            ]

        label = "🎯 Matched Jobs" if show_match else "💼 Extracted Jobs"
        st.subheader(f"{label} ({len(jobs_to_show)})")
        if show_match:
            st.caption("Sorted by match score — highest first")

        for i, job in enumerate(jobs_to_show):
            _render_job_card(job, i, show_match=show_match)

    elif not extracted:
        st.info(
            f"Click **🚀 Extract Jobs** above to process "
            f"{len(job_emails)} recruitment emails."
        )
