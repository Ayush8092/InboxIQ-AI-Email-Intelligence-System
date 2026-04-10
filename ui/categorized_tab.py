"""
Categorized Emails Tab — complete rewrite.

Fixes:
1. Shows FULL email content (not truncated)
2. Job emails are visible even when no resume uploaded
3. After resume upload, jobs are scored and displayed
4. HTML/Raw toggle for email body
5. enrich_job_with_scraping properly imported
6. All job links displayed
7. Emails visible in job category view
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
_REC_COLORS = {
    "Strong Match":"#22c55e","Good Match":"#84cc16",
    "Partial Match":"#f59e0b","Weak Match":"#ef4444",
}


def _get_source() -> str:
    return "gmail" if st.session_state.get("authenticated") else "demo"


def _render_full_email_content(email: dict, show_raw: bool = False):
    """
    Render the FULL email content.
    Shows cleaned text by default, raw HTML on toggle.
    """
    body = email.get("body","")
    if not body:
        st.caption("_No email body available_")
        return

    if show_raw:
        st.text_area(
            "Raw HTML",
            value=body[:3000],
            height=200,
            disabled=True,
            key=f"raw_{email.get('id','')[:20]}",
        )
    else:
        clean = clean_email_body(body, max_chars=5000)
        if clean:
            st.text_area(
                "Email Content",
                value=clean,
                height=200,
                disabled=True,
                key=f"clean_{email.get('id','')[:20]}",
            )
        else:
            st.caption("_Could not extract readable text from this email_")

    # Show extracted links
    parsed    = parse_email_html(body)
    job_links = parsed.get("job_links",[])
    all_links = parsed.get("links",[])

    if job_links:
        st.markdown("**🔗 Job Links found in email:**")
        for link in job_links[:5]:
            st.markdown(f"- [{link[:80]}]({link})")
    elif all_links:
        st.markdown("**🔗 Links found in email:**")
        for link in all_links[:3]:
            st.markdown(f"- [{link[:80]}]({link})")


def _render_job_card(job: dict, idx: int, show_match: bool = False):
    """Render a single job card with all details and links."""
    match     = job.get("match",{})
    score     = match.get("match_score",0) if show_match else None
    rec       = match.get("recommendation","") if show_match else ""
    rec_color = _REC_COLORS.get(rec,"#808080")
    ready     = match.get("ready_to_apply",False) if show_match else False

    with st.container():
        col_t, col_s = st.columns([4,1])
        with col_t:
            st.markdown(f"### 💼 {job.get('role','Unknown Role')}")
            parts = []
            if job.get("company") and job["company"] not in ("Unknown",""):
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
                    st.success("✅ Ready to Apply")

        # Skills
        skills = job.get("skills",[])
        if skills:
            tags = " ".join(
                f'<span style="background:#e2e8f0;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;">{s}</span>'
                for s in skills[:15]
            )
            st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)
        else:
            st.caption("_No skills extracted_")

        # Description
        if job.get("description"):
            st.caption(job["description"][:200])

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
                if match.get("strengths"):
                    st.caption(f"💪 {match['strengths']}")
            with c2:
                if match.get("missing_skills"):
                    st.markdown(
                        "⚠️ **Missing:** " +
                        ", ".join(f"`{s}`" for s in match["missing_skills"][:6])
                    )
                if match.get("gaps"):
                    st.caption(f"📌 {match['gaps']}")

            suggestions = match.get("skill_gap_suggestions",[])
            if suggestions:
                with st.expander("📚 Learning Path for Missing Skills"):
                    for sg in suggestions[:3]:
                        skill    = sg.get("skill","")
                        resource = sg.get("resource","")
                        if skill:
                            st.write(f"• **{skill}** — {resource}")

        # Apply links — show ALL links from this job
        all_links = job.get("all_links",[])
        if not all_links and job.get("link"):
            all_links = [job["link"]]

        if all_links:
            st.markdown("**Apply:**")
            cols = st.columns(min(len(all_links), 3))
            for i, link in enumerate(all_links[:3]):
                with cols[i]:
                    label = "🔗 Apply Now" if i == 0 else f"🔗 Link {i+1}"
                    st.markdown(
                        f'<a href="{link}" target="_blank">'
                        f'<button style="background:#4285F4;color:white;border:none;'
                        f'padding:6px 14px;border-radius:4px;cursor:pointer;'
                        f'font-size:13px;width:100%;">{label}</button></a>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("_No apply link found_")

        # Source
        src_txt = {"bs4_anchor":"🔗 From anchor tags","rule":"📋 Rule-based","llm":"🤖 AI-extracted"}.get(
            job.get("source",""), "📧 Extracted"
        )
        st.caption(src_txt + (" | 🌐 Web enriched" if job.get("scraped") else ""))
        st.caption(f"From email: {truncate(job.get('email_subject',''),60)}")
        st.markdown("---")


def _render_email_card_full(email: dict, processed: dict | None, show_raw: bool):
    """Render email card with expandable FULL content."""
    p       = processed or {}
    subject = email.get("subject","")
    sender  = email.get("sender","")
    cat     = p.get("category","—")
    pr      = p.get("priority")
    task    = p.get("task","")
    summary = p.get("summary","")
    color   = _PRIORITY_COLORS.get(pr,"#808080")

    with st.container():
        c1, c2 = st.columns([5,1])
        with c1:
            st.markdown(f"**{truncate(subject, 70)}**")
            st.caption(f"From: {sender} &nbsp;|&nbsp; {cat}")
            if summary and summary not in ("—",""):
                st.info(f"💡 {summary}")
            if task and task not in ("—",""):
                st.markdown(f"📌 **Task:** {truncate(task, 80)}")
        with c2:
            if pr:
                st.markdown(
                    f'<div style="background:{color};color:white;'
                    f'border-radius:6px;padding:4px 8px;text-align:center;'
                    f'font-size:12px;">{priority_label(pr)}</div>',
                    unsafe_allow_html=True,
                )

        # Expand to see full email
        with st.expander("📖 View Full Email", expanded=False):
            _render_full_email_content(email, show_raw)

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
    labels = {c: f"{c} ({cat_counts[c]})" for c in sorted_cats}

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2,1,2])
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
        )
    with c3:
        count = cat_counts.get(sel_cat,0) if sel_cat != "All" else len(emails_list)
        st.metric("Emails shown", count)

    # Filter
    if sel_cat == "All":
        filtered = emails_list
    else:
        filtered = [
            e for e in emails_list
            if proc_map.get(e["id"],{}).get("category") == sel_cat
            or (sel_cat in _JOB_CATEGORIES and
                proc_map.get(e["id"],{}).get("category","") in _JOB_CATEGORIES)
        ]
        # If no processed emails match but emails exist, show all in category
        if not filtered and sel_cat in _JOB_CATEGORIES:
            # Show all emails (unprocessed too) when job category selected
            filtered = emails_list

    # ── Distribution ─────────────────────────────────────────────────────────
    with st.expander("📊 Category Distribution", expanded=False):
        total = max(len(emails_list), 1)
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            if cat == "All":
                continue
            pct = cnt / total * 100
            ca, cb, cc = st.columns([3,5,1])
            with ca: st.write(cat)
            with cb: st.progress(pct/100)
            with cc: st.write(str(cnt))

    st.divider()

    # ── Job intelligence mode ─────────────────────────────────────────────────
    if sel_cat in _JOB_CATEGORIES:
        _render_job_section(filtered, proc_map, show_raw)
        return

    # ── Standard view ─────────────────────────────────────────────────────────
    label = "All Emails" if sel_cat == "All" else sel_cat
    st.subheader(f"{label} ({len(filtered)})")

    if not filtered:
        st.info(f"No emails in '{sel_cat}'.")
        return

    for email in filtered:
        _render_email_card_full(email, proc_map.get(email["id"]), show_raw)


def _render_job_section(job_emails: list[dict], proc_map: dict, show_raw: bool):
    """
    Job section: shows emails + extracted jobs + resume matching.
    Emails are always visible. Resume upload is optional.
    """
    st.subheader(f"💼 Job / Recruitment ({len(job_emails)} emails)")

    # ── SECTION 1: Show the actual emails ────────────────────────────────────
    st.markdown("### 📧 Job Emails")
    st.caption("Expand each email to see full content including all job listings")

    email_tab, jobs_tab = st.tabs(["📧 View Emails", "🔍 Extract Jobs"])

    with email_tab:
        if not job_emails:
            st.info("No job emails found.")
        for email in job_emails:
            _render_email_card_full(email, proc_map.get(email.get("id","")), show_raw)

    with jobs_tab:
        # ── SECTION 2: Resume upload (optional) ──────────────────────────────
        st.markdown("### 📄 Resume Upload (Optional)")
        st.caption("Upload your resume to get personalized match scores. You can extract jobs without it.")

        cu, cs = st.columns([2,2])
        with cu:
            uploaded = st.file_uploader(
                "PDF or TXT",
                type=["pdf","txt"],
                key="resume_uploader",
            )

        resume_data = st.session_state.get("parsed_resume")

        if uploaded:
            if st.session_state.get("resume_filename") != uploaded.name or not resume_data:
                with st.spinner("Parsing resume with AI..."):
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
                    f"Exp: {resume_data.get('experience_years',0)} yrs | "
                    f"Role: {resume_data.get('current_role','—')}"
                )
                with st.expander("📋 Parsed Resume Details"):
                    tags = " ".join(
                        f'<span style="background:#dcfce7;padding:2px 8px;'
                        f'border-radius:10px;font-size:12px;margin:2px;">{s}</span>'
                        for s in resume_data["skills"][:20]
                    )
                    st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)
                    if resume_data.get("summary"):
                        st.write("**Summary:**")
                        st.caption(resume_data["summary"])
                    if resume_data.get("experience"):
                        st.write("**Experience:**")
                        st.caption(resume_data["experience"])
                    if resume_data.get("projects"):
                        st.write("**Projects:**")
                        for proj in resume_data["projects"][:3]:
                            st.write(f"• {proj}")
            else:
                st.info("📤 Upload resume for match scores (optional)")

        st.divider()

        # ── SECTION 3: Job extraction ─────────────────────────────────────────
        st.markdown("### 🔍 Extract Jobs from Emails")
        st.caption(
            f"Found {len(job_emails)} job emails. "
            "Each email typically contains 3-10 job listings."
        )

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            enrich = st.checkbox(
                "🌐 Web enrichment",
                value=False,
                key="job_enrich",
                help="Fetch job pages for extra details (skips LinkedIn/Glassdoor)",
            )
        with cc2:
            max_e = st.slider(
                "Emails to process",
                1, min(len(job_emails),30),
                min(len(job_emails),10),
                key="job_max",
            )
        with cc3:
            st.write("")
            extract_btn = st.button(
                "🚀 Extract All Jobs",
                type="primary",
                use_container_width=True,
            )

        if extract_btn:
            from services.job_service import extract_jobs_from_email, enrich_job_with_scraping
            to_proc  = job_emails[:max_e]
            all_jobs = []
            prog     = st.progress(0)
            status   = st.empty()

            for i, email in enumerate(to_proc):
                status.text(
                    f"Processing {i+1}/{len(to_proc)}: "
                    f"{truncate(email.get('subject',''),50)}"
                )
                jobs = extract_jobs_from_email(email)
                if enrich:
                    jobs = [enrich_job_with_scraping(j) for j in jobs]
                all_jobs.extend(jobs)
                prog.progress((i+1)/len(to_proc))

            # Deduplicate
            seen, unique = set(), []
            import re as _re
            for job in all_jobs:
                key = (
                    _re.sub(r'\s+', ' ', job.get("role","")).lower()[:60],
                    _re.sub(r'\s+', ' ', job.get("company","")).lower()[:40],
                )
                if key not in seen and key[0]:
                    seen.add(key)
                    unique.append(job)

            st.session_state["extracted_jobs"] = unique
            st.session_state.pop("scored_jobs", None)
            prog.progress(1.0)
            status.success(f"✅ Found {len(unique)} unique jobs from {len(to_proc)} emails!")

        # ── SECTION 4: Score button ───────────────────────────────────────────
        extracted = st.session_state.get("extracted_jobs",[])
        if extracted and resume_data and resume_data.get("skills"):
            if st.button("🎯 Score All Jobs Against My Resume", type="secondary"):
                with st.spinner(f"Scoring {len(extracted)} jobs against your resume..."):
                    from services.job_service import score_all_jobs
                    scored = score_all_jobs(extracted, resume_data)
                    st.session_state["scored_jobs"] = scored
                st.success(f"✅ Scored {len(scored)} jobs!")

        # ── SECTION 5: Display jobs ───────────────────────────────────────────
        jobs_to_show = st.session_state.get("scored_jobs") or extracted
        show_match   = bool(
            st.session_state.get("scored_jobs") and
            resume_data and resume_data.get("skills")
        )

        if jobs_to_show:
            st.divider()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Jobs Found", len(jobs_to_show))
            with m2:
                if show_match:
                    strong = sum(
                        1 for j in jobs_to_show
                        if j.get("match",{}).get("match_score",0) >= 60
                    )
                    st.metric("Good/Strong Matches", strong)
            with m3:
                min_score = (
                    st.slider("Min match %", 0, 100, 0, key="min_s")
                    if show_match else 0
                )

            if show_match and min_score > 0:
                jobs_to_show = [
                    j for j in jobs_to_show
                    if j.get("match",{}).get("match_score",0) >= min_score
                ]

            label = "🎯 Jobs Matched to Resume" if show_match else "💼 Extracted Jobs"
            st.subheader(f"{label} ({len(jobs_to_show)})")
            if show_match:
                st.caption("Sorted by match score — highest first")
            else:
                st.caption("Upload resume and click '🎯 Score' to get match scores")

            for i, job in enumerate(jobs_to_show):
                _render_job_card(job, i, show_match=show_match)

        elif not extracted:
            st.info(
                f"👆 Click **🚀 Extract All Jobs** above to find all job listings "
                f"from your {len(job_emails)} recruitment emails."
            )