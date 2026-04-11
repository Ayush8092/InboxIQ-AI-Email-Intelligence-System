"""
Job Task Tab — new dedicated tab for resume upload and job matching.

Features:
1. Resume upload (PDF/TXT) with section-aware parsing
2. Name extracted from document header only
3. Experience from experience/internship section only
4. Skills from skills/projects section only
5. LLM-powered extraction for each section
6. Job extraction from job emails using LLM
7. Semantic matching with scoring
8. Public company info via LLM for context
"""
import re
import json
import time
import streamlit as st
from functools import lru_cache
from utils.secure_logger import get_secure_logger
from utils.email_cleaner import clean_email_body, parse_email_html

logger = get_secure_logger(__name__)


# ── Cached LLM ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=300)
def _llm(prompt: str) -> str:
    from utils.llm_client import call_llm
    return call_llm(prompt, temperature=0.0, max_tokens=2000, use_cache=True)


def _llm_json(prompt: str, default) -> dict | list:
    try:
        raw = _llm(prompt)
        m   = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        logger.warning(f"LLM JSON parse failed: {type(e).__name__}")
    return default


# ── PDF text extraction ────────────────────────────────────────────────────────

def _extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    try:
        import pdfminer.high_level as pmh, io
        text = pmh.extract_text(io.BytesIO(file_bytes)) or ""
    except Exception:
        pass
    if len(text.strip()) < 100:
        try:
            import pypdf, io
            r    = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(p.extract_text() or "" for p in r.pages)
        except Exception:
            pass
    if len(text.strip()) < 100:
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    return text.strip()


def _clean_text(raw: str) -> str:
    text = raw.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Section-based resume parsing ──────────────────────────────────────────────

_SECTION_HEADERS = {
    "name_area":    r'^[A-Z][^\n]{3,60}$',
    "summary":      r'(?:professional\s+)?summary|objective|profile|about',
    "education":    r'education|academic|qualification|degree',
    "experience":   r'experience|work\s+experience|employment|internship|internships|professional\s+experience',
    "skills":       r'technical\s+skills?|skills?|core\s+competencies|technologies|tech\s+stack',
    "projects":     r'projects?|project\s+work|personal\s+projects|academic\s+projects',
    "certifications": r'certifications?|certificates?|achievements?|awards?',
}


def _split_into_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections.
    Returns {section_name: content}
    """
    sections: dict[str, str] = {}
    lines = text.split('\n')

    # First line(s) are likely the name/header
    name_lines = []
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 60 and not re.search(
            r'@|\.com|phone|email|linkedin|github', line, re.IGNORECASE
        ):
            name_lines.append(line)
            if len(name_lines) >= 2:
                break
    sections['header'] = '\n'.join(name_lines)

    current_section = 'intro'
    current_content = []
    sections[current_section] = ''

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_content.append('')
            continue

        # Check if this line is a section header
        matched_section = None
        for sec_key, pattern in _SECTION_HEADERS.items():
            if sec_key == 'name_area':
                continue
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 50:
                matched_section = sec_key
                break

        if matched_section:
            # Save current section
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = matched_section
            current_content = []
        else:
            current_content.append(line)

    sections[current_section] = '\n'.join(current_content).strip()
    return sections


# ── LLM prompts for section-based parsing ────────────────────────────────────

_NAME_PROMPT = """\
Extract the person's full name from this resume header.
Return ONLY the full name, nothing else.
If not found, return "Unknown".

Header text:
{header}"""

_SKILLS_PROMPT = """\
Extract ALL technical skills from this section.
Return JSON array of skill names only.
Skills must be real technologies (python, java, sql, react, etc.)
Minimum 3 characters each.

Section:
{section}

Return: ["skill1", "skill2", "skill3"]"""

_EXPERIENCE_PROMPT = """\
Extract work experience from this section.
Return JSON:
{{
  "experience_years": 1,
  "current_role": "most recent job title",
  "experience": "2-3 sentence summary with dates and companies",
  "roles": [
    {{"title": "role", "company": "company", "duration": "dates", "description": "what you did"}}
  ]
}}

Section:
{section}

Return JSON only:"""

_PROJECTS_PROMPT = """\
Extract projects from this section.
Return JSON array:
[{{"name": "project name", "description": "what it does", "technologies": ["tech1", "tech2"]}}]

Section:
{section}

Return JSON array only:"""

_EDUCATION_PROMPT = """\
Extract education from this section.
Return: "Degree, Field, Institution, Year"

Section:
{section}

Return single line only:"""


def parse_resume_sections(file_bytes: bytes, file_type: str = "pdf") -> dict:
    """
    Parse resume using section detection + targeted LLM calls per section.
    Name ← header section only
    Experience ← experience/internship section only
    Skills ← skills + projects sections only
    """
    if file_type == "pdf":
        raw = _extract_pdf_text(file_bytes)
    else:
        raw = file_bytes.decode("utf-8", errors="ignore")

    raw = _clean_text(raw)

    if len(raw.strip()) < 100:
        return _empty_resume()

    sections = _split_into_sections(raw)
    result   = _empty_resume()
    result["raw_text_length"] = len(raw)

    # 1. Extract name from header ONLY
    header = sections.get("header","") or raw.split('\n')[0]
    name   = _llm(f"Extract the person's full name from this text. Return ONLY the full name:\n{header}").strip()
    name   = re.sub(r'[^A-Za-z\s]','',name).strip()
    if len(name) < 2 or len(name) > 60:
        # fallback: first non-empty line that looks like a name
        for line in raw.split('\n')[:5]:
            line = line.strip()
            if (2 < len(line) < 50 and
                not re.search(r'[@\d\|/\\]', line) and
                re.search(r'[A-Z][a-z]', line)):
                name = line
                break
    result["name"] = name

    # 2. Extract experience from experience/internship section ONLY
    exp_section = (
        sections.get("experience","") or
        sections.get("internships","") or
        sections.get("internship","") or ""
    )
    if exp_section and len(exp_section) > 20:
        exp_data = _llm_json(
            _EXPERIENCE_PROMPT.format(section=exp_section[:2000]),
            {}
        )
        if isinstance(exp_data, dict):
            result["experience_years"] = int(exp_data.get("experience_years") or 0)
            result["current_role"]     = str(exp_data.get("current_role",""))
            result["experience"]       = str(exp_data.get("experience",""))
            result["roles"]            = exp_data.get("roles",[]) or []

    # 3. Extract skills from skills section AND projects section
    skills_section   = sections.get("skills","") or sections.get("core competencies","") or ""
    projects_section = sections.get("projects","") or sections.get("project work","") or ""

    all_skills = []

    if skills_section and len(skills_section) > 10:
        sk = _llm_json(
            _SKILLS_PROMPT.format(section=skills_section[:1500]),
            []
        )
        if isinstance(sk, list):
            all_skills.extend(sk)

    if projects_section and len(projects_section) > 10:
        # Extract tech from projects
        sk2 = _llm_json(
            f"Extract all technologies/skills mentioned in this projects section.\n"
            f"Return JSON array of technology names only:\n{projects_section[:1000]}\n"
            f'Return: ["tech1","tech2"]',
            []
        )
        if isinstance(sk2, list):
            all_skills.extend(sk2)
        # Also parse projects list
        proj = _llm_json(
            _PROJECTS_PROMPT.format(section=projects_section[:1500]),
            []
        )
        if isinstance(proj, list):
            result["projects"] = [
                f"{p.get('name','')}: {p.get('description','')}"
                for p in proj[:5]
            ]

    result["skills"] = _validate_skills(all_skills)

    # 4. Education
    edu_section = sections.get("education","") or sections.get("academic","") or ""
    if edu_section:
        edu = _llm(
            f"Extract education as: 'Degree, Field, Institution, Year'\n"
            f"Section:\n{edu_section[:500]}\nReturn single line:"
        ).strip()
        result["education"] = edu[:200]

    # 5. Certifications
    cert_section = sections.get("certifications","") or sections.get("achievements","") or ""
    if cert_section:
        certs = _llm_json(
            f"Extract certifications as JSON array of strings:\n{cert_section[:500]}\nReturn: [\"cert1\"]",
            []
        )
        if isinstance(certs, list):
            result["certifications"] = [str(c) for c in certs[:5]]

    # 6. Generate summary from all sections
    summary_input = f"""
Name: {result['name']}
Current Role: {result['current_role']}
Experience: {result['experience']}
Skills: {', '.join(result['skills'][:15])}
Education: {result['education']}
""".strip()
    summary = _llm(
        f"Write a 2-sentence professional summary for this candidate:\n{summary_input}\n"
        f"Return only the 2-sentence summary:"
    ).strip()
    result["summary"] = summary

    return result


def _validate_skills(skills: list) -> list[str]:
    _VALID = {
        "python","java","javascript","typescript","c++","c#","golang","go","rust",
        "swift","kotlin","php","ruby","scala","r programming","matlab","bash",
        "shell scripting","flutter","react","reactjs","angular","vue","nextjs",
        "nodejs","express","django","flask","fastapi","spring boot","html","css",
        "tailwind","bootstrap","graphql","rest api","machine learning",
        "deep learning","nlp","natural language processing","computer vision",
        "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","scipy",
        "sql","mysql","postgresql","mongodb","redis","elasticsearch","kafka",
        "spark","hadoop","airflow","tableau","power bi","excel","data analysis",
        "data engineering","mlops","llm","openai","langchain","rag","aws","azure",
        "gcp","docker","kubernetes","ci/cd","jenkins","github actions","terraform",
        "linux","git","devops","microservices","system design","agile","xgboost",
        "lightgbm","transformers","bert","hugging face","generative ai",
        "statistics","probability","data visualization","feature engineering",
        "model deployment","a/b testing","time series","reinforcement learning",
    }
    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str): continue
        s = s.strip().strip("\"'.,;").strip()
        if len(s) < 2 or len(s) > 60: continue
        if re.match(r'^[\d\s\W]+$', s): continue
        if re.match(r'^[a-zA-Z]$', s): continue
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            cleaned.append(s)
    return cleaned[:25]


def _empty_resume() -> dict:
    return {
        "name":"","skills":[],"experience_years":0,"experience":"",
        "current_role":"","education":"","projects":[],"certifications":[],
        "summary":"","raw_text_length":0,"roles":[],
    }


# ── Job extraction from emails ─────────────────────────────────────────────────

_JOB_EXTRACT_PROMPT = """\
This is a job alert email. Extract ALL job listings.

Subject: {subject}
Email content:
{body}

Find EVERY job mentioned. Each job has: title, company, location, salary, skills.
Job titles are usually links (anchor text) or bold text.

Return JSON array — extract ALL jobs you can find (typically 5-12 per email):
[
  {{
    "role": "Data Analyst",
    "company": "Terrier Security Services",
    "location": "India",
    "salary": "₹20K - ₹42K (Employer Est.)",
    "skills": ["statistics","microsoft excel","data analysis"],
    "description": "Data analyst role focusing on..."
  }}
]

Return [] only if truly no jobs found.
JSON array only, no explanation:"""


def extract_jobs_from_emails_llm(emails: list[dict]) -> list[dict]:
    """
    Extract all jobs from job emails using LLM.
    Processes each email and finds all job listings.
    """
    all_jobs = []
    for email in emails:
        body  = email.get("body","")
        clean = clean_email_body(body, max_chars=4000)

        if not clean:
            continue

        prompt = _JOB_EXTRACT_PROMPT.format(
            subject=email.get("subject",""),
            body=clean[:3500],
        )

        jobs_data = _llm_json(prompt, [])
        if not isinstance(jobs_data, list):
            continue

        # Also extract links from this email
        parsed    = parse_email_html(body)
        job_links = parsed.get("job_links",[])
        all_lnks  = parsed.get("links",[])
        links     = job_links or all_lnks

        for i, job in enumerate(jobs_data):
            if not isinstance(job, dict):
                continue
            role = str(job.get("role","")).strip()
            if len(role) < 3:
                continue

            skills = _validate_skills(job.get("skills",[]) or [])
            link   = links[i] if i < len(links) else None

            all_jobs.append({
                "role":          role[:100],
                "company":       str(job.get("company","Unknown")).strip()[:100],
                "location":      str(job.get("location","Not specified")).strip()[:100],
                "salary":        str(job.get("salary","Not specified")).strip()[:100],
                "skills":        skills,
                "link":          link,
                "all_links":     [link] if link else [],
                "description":   str(job.get("description","")).strip()[:200],
                "email_id":      email.get("id",""),
                "email_subject": email.get("subject",""),
                "source":        "llm",
            })

        time.sleep(0.3)  # rate limit

    # Deduplicate
    seen, unique = set(), []
    for job in all_jobs:
        key = (job.get("role","").lower()[:60], job.get("company","").lower()[:40])
        if key not in seen and key[0]:
            seen.add(key)
            unique.append(job)

    return unique


# ── Job-Resume matching with company context ──────────────────────────────────

_MATCH_PROMPT = """\
Match this job with the candidate's resume. Be realistic and specific.

JOB:
Title: {role}
Company: {company}
Location: {location}
Required Skills: {job_skills}
Description: {description}

CANDIDATE:
Name: {name}
Skills: {resume_skills}
Experience: {experience_years} years — {experience}
Current Role: {current_role}
Education: {education}
Projects: {projects}

Score this match (0-97, NEVER 100):
- Skill overlap: 50 points max
- Role/domain fit: 30 points max  
- Experience level: 20 points max

Return JSON:
{{
  "match_score": 68,
  "matched_skills": ["python","sql"],
  "missing_skills": ["kubernetes"],
  "fit_reason": "You have strong Python and ML skills which align with this Data Analyst role at {company}",
  "strengths": "specific strength 1, strength 2",
  "gaps": "missing skill X, need more Y experience",
  "recommendation": "Good Match",
  "ready_to_apply": true,
  "skill_gap_suggestions": [
    {{"skill": "kubernetes", "resource": "Kubernetes on Coursera (free audit)"}}
  ]
}}

recommendation must be: Strong Match (80+), Good Match (60-79), Partial Match (40-59), Weak Match (<40)
JSON only:"""


def score_jobs_against_resume(
    jobs: list[dict],
    resume: dict,
) -> list[dict]:
    """Score all jobs against resume using LLM semantic matching."""
    scored = []
    resume_skills = set(s.lower() for s in resume.get("skills",[]))

    for job in jobs:
        job_skills = set(s.lower() for s in _validate_skills(job.get("skills",[])))
        matched    = sorted(job_skills & resume_skills)
        missing    = sorted(job_skills - resume_skills)

        # Rule-based baseline
        if job_skills:
            overlap    = (len(matched) / len(job_skills)) * 50
            rule_score = min(97, int(overlap + 20))
        else:
            rule_score = 35

        def _rec(s):
            if s >= 80: return "Strong Match"
            if s >= 60: return "Good Match"
            if s >= 40: return "Partial Match"
            return "Weak Match"

        try:
            prompt = _MATCH_PROMPT.format(
                role=job.get("role",""),
                company=job.get("company",""),
                location=job.get("location",""),
                job_skills=", ".join(list(job_skills)[:12]),
                description=job.get("description",""),
                name=resume.get("name",""),
                resume_skills=", ".join(list(resume_skills)[:20]),
                experience_years=resume.get("experience_years",0),
                experience=resume.get("experience","")[:200],
                current_role=resume.get("current_role",""),
                education=resume.get("education",""),
                projects=", ".join(resume.get("projects",[])[:3]),
            )
            time.sleep(0.2)
            data = _llm_json(prompt, {})
            if isinstance(data, dict) and data.get("match_score") is not None:
                score = max(0, min(97, int(data.get("match_score", rule_score))))
                match = {
                    "match_score":           score,
                    "matched_skills":        _validate_skills(data.get("matched_skills",matched)),
                    "missing_skills":        _validate_skills(data.get("missing_skills",missing)),
                    "fit_reason":            str(data.get("fit_reason","")),
                    "strengths":             str(data.get("strengths","")),
                    "gaps":                  str(data.get("gaps","")),
                    "recommendation":        str(data.get("recommendation",_rec(score))),
                    "ready_to_apply":        bool(data.get("ready_to_apply",score>=55)),
                    "skill_gap_suggestions": data.get("skill_gap_suggestions",[]) or [],
                }
            else:
                raise ValueError("No match_score in response")

        except Exception:
            match = {
                "match_score":           rule_score,
                "matched_skills":        list(matched),
                "missing_skills":        list(missing),
                "fit_reason":            f"{len(matched)}/{len(job_skills)} required skills matched",
                "strengths":             f"Matched skills: {', '.join(list(matched)[:4])}",
                "gaps":                  f"Missing: {', '.join(list(missing)[:4])}",
                "recommendation":        _rec(rule_score),
                "ready_to_apply":        rule_score >= 55,
                "skill_gap_suggestions": [
                    {"skill":s,"resource":f"Search '{s}' on Coursera"}
                    for s in list(missing)[:2]
                ],
            }

        scored.append({**job, "match": match})

    scored.sort(key=lambda x: x.get("match",{}).get("match_score",0), reverse=True)
    return scored


# ── UI rendering ──────────────────────────────────────────────────────────────

_REC_COLORS = {
    "Strong Match":"#22c55e","Good Match":"#84cc16",
    "Partial Match":"#f59e0b","Weak Match":"#ef4444",
}


def _render_job_card(job: dict, idx: int, show_match: bool):
    match     = job.get("match",{})
    score     = match.get("match_score",0) if show_match else None
    rec       = match.get("recommendation","") if show_match else ""
    rec_color = _REC_COLORS.get(rec,"#808080")
    ready     = match.get("ready_to_apply",False) if show_match else False

    with st.container():
        ct, cs = st.columns([4,1])
        with ct:
            st.markdown(f"#### 💼 {job.get('role','Unknown')}")
            parts = []
            co = job.get("company","")
            lo = job.get("location","")
            sa = job.get("salary","")
            if co and co != "Unknown": parts.append(f"🏢 **{co}**")
            if lo and lo != "Not specified": parts.append(f"📍 {lo}")
            if sa and sa != "Not specified": parts.append(f"💰 {sa}")
            if parts:
                st.markdown(" &nbsp;|&nbsp; ".join(parts))

        with cs:
            if show_match and score is not None:
                st.markdown(
                    f'<div style="background:{rec_color};color:white;'
                    f'border-radius:8px;padding:8px;text-align:center;'
                    f'font-weight:bold;font-size:22px;">{score}%</div>'
                    f'<div style="text-align:center;font-size:11px;'
                    f'color:{rec_color};font-weight:bold;margin-top:4px">{rec}</div>',
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
                for s in skills[:12]
            )
            st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)

        if job.get("description"):
            st.caption(job["description"][:200])

        # Match details
        if show_match and match:
            c1, c2 = st.columns(2)
            with c1:
                if match.get("matched_skills"):
                    st.markdown("✅ **Matched:** " + ", ".join(f"`{s}`" for s in match["matched_skills"][:5]))
                if match.get("fit_reason"):
                    st.caption(f"💡 {match['fit_reason']}")
            with c2:
                if match.get("missing_skills"):
                    st.markdown("⚠️ **Missing:** " + ", ".join(f"`{s}`" for s in match["missing_skills"][:5]))
                if match.get("gaps"):
                    st.caption(f"📌 {match['gaps']}")

            sgs = match.get("skill_gap_suggestions",[])
            if sgs:
                with st.expander("📚 Learning Path"):
                    for sg in sgs[:3]:
                        st.write(f"• **{sg.get('skill','')}** — {sg.get('resource','')}")

        # Apply links
        links = job.get("all_links",[]) or ([job["link"]] if job.get("link") else [])
        if links:
            cols = st.columns(min(len(links),3))
            for i, lnk in enumerate(links[:3]):
                with cols[i]:
                    label = "🔗 Apply Now" if i==0 else f"🔗 Link {i+1}"
                    st.markdown(
                        f'<a href="{lnk}" target="_blank">'
                        f'<button style="background:#4285F4;color:white;border:none;'
                        f'padding:6px 12px;border-radius:4px;cursor:pointer;width:100%">'
                        f'{label}</button></a>',
                        unsafe_allow_html=True,
                    )
        st.markdown("---")


def render_job_task_tab():
    """Main render function for the Job Task tab."""
    st.header("💼 Job Task — Resume & Job Matching")

    # ── Step 1: Upload Resume ─────────────────────────────────────────────────
    st.subheader("📄 Step 1: Upload Your Resume")
    st.caption("Your resume is parsed section by section for accuracy")

    cu, cs = st.columns([2,2])
    with cu:
        uploaded = st.file_uploader(
            "Upload Resume (PDF or TXT)",
            type=["pdf","txt"],
            key="job_task_resume",
        )

    resume = st.session_state.get("job_task_resume_data")

    if uploaded:
        if st.session_state.get("job_task_resume_name") != uploaded.name or not resume:
            with st.spinner("Parsing resume section by section..."):
                fb     = uploaded.read()
                ft     = "pdf" if uploaded.name.lower().endswith(".pdf") else "txt"
                resume = parse_resume_sections(fb, ft)
                st.session_state["job_task_resume_data"] = resume
                st.session_state["job_task_resume_name"] = uploaded.name

    with cs:
        if resume and resume.get("name"):
            st.success(f"✅ **{resume['name']}**")
            st.caption(
                f"Skills: {len(resume.get('skills',[]))} | "
                f"Exp: {resume.get('experience_years',0)} yrs | "
                f"Role: {resume.get('current_role','—')}"
            )
        elif resume:
            st.warning("⚠️ Resume parsed but some fields missing")
        else:
            st.info("📤 Upload your resume to get started")

    if resume and resume.get("name"):
        with st.expander("📋 View Parsed Resume Details", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Name:** {resume.get('name','—')}")
                st.write(f"**Current Role:** {resume.get('current_role','—')}")
                st.write(f"**Experience:** {resume.get('experience_years',0)} years")
                st.write(f"**Education:** {resume.get('education','—')}")
                if resume.get("experience"):
                    st.write("**Experience Summary:**")
                    st.caption(resume["experience"])
            with c2:
                if resume.get("skills"):
                    tags = " ".join(
                        f'<span style="background:#dcfce7;padding:2px 8px;'
                        f'border-radius:10px;font-size:12px;margin:2px;">{s}</span>'
                        for s in resume["skills"][:20]
                    )
                    st.markdown(f"**Skills ({len(resume['skills'])}):**")
                    st.markdown(tags, unsafe_allow_html=True)
                if resume.get("projects"):
                    st.write("**Projects:**")
                    for p in resume["projects"][:3]:
                        st.caption(f"• {p[:100]}")
                if resume.get("certifications"):
                    st.write("**Certifications:**")
                    for c in resume["certifications"][:3]:
                        st.caption(f"• {c}")
                if resume.get("summary"):
                    st.write("**Summary:**")
                    st.caption(resume["summary"])

    st.divider()

    # ── Step 2: Extract Jobs from Emails ─────────────────────────────────────
    st.subheader("🔍 Step 2: Extract Jobs from Your Emails")

    source      = "gmail" if st.session_state.get("authenticated") else "demo"
    all_emails  = []
    try:
        from memory.repository import get_all_emails, get_all_processed
        all_emails = get_all_emails(source=source)
        proc_list  = get_all_processed(source=source)
        proc_map   = {p["email_id"]: p for p in proc_list}
    except Exception as e:
        st.error(f"Could not load emails: {type(e).__name__}")
        return

    # Find job emails
    _JOB_SENDERS = ["glassdoor","naukri","linkedin","indeed","monster","resume.io","jobs@","career@"]
    job_emails   = []
    for email in all_emails:
        cat    = proc_map.get(email.get("id",""),{}).get("category","")
        sender = email.get("sender","").lower()
        if cat in {"Job / Recruitment","Job","job_recruitment"} or \
           any(js in sender for js in _JOB_SENDERS):
            job_emails.append(email)

    if not job_emails:
        st.info(
            f"No job emails found yet. "
            f"Found {len(all_emails)} total emails. "
            "Make sure you have processed emails in the Inbox tab first."
        )
    else:
        st.success(f"Found **{len(job_emails)} job emails** to extract from")

        c1, c2, c3 = st.columns(3)
        with c1:
            max_emails = st.slider(
                "Emails to process",
                1, min(len(job_emails),20),
                min(len(job_emails),5),
                key="jt_max_emails",
            )
        with c2:
            st.write("")
            extract_btn = st.button(
                "🚀 Extract All Jobs",
                type="primary",
                use_container_width=True,
                key="jt_extract",
            )
        with c3:
            st.write("")
            if st.button("🗑️ Clear Results", use_container_width=True, key="jt_clear"):
                st.session_state.pop("job_task_jobs", None)
                st.session_state.pop("job_task_scored", None)
                st.rerun()

        if extract_btn:
            to_proc  = job_emails[:max_emails]
            prog     = st.progress(0)
            status   = st.empty()
            status.text(f"Extracting jobs from {len(to_proc)} emails using AI...")

            extracted = extract_jobs_from_emails_llm(to_proc)

            prog.progress(1.0)
            st.session_state["job_task_jobs"]   = extracted
            st.session_state.pop("job_task_scored", None)
            status.success(
                f"✅ Found **{len(extracted)} unique jobs** from {len(to_proc)} emails!"
            )

    st.divider()

    # ── Step 3: Score Jobs ────────────────────────────────────────────────────
    extracted = st.session_state.get("job_task_jobs",[])

    if extracted:
        st.subheader(f"📊 Step 3: Jobs Found ({len(extracted)})")

        if resume and resume.get("skills"):
            if not st.session_state.get("job_task_scored"):
                if st.button(
                    f"🎯 Score {len(extracted)} Jobs Against My Resume",
                    type="secondary",
                    key="jt_score",
                ):
                    with st.spinner("Scoring jobs with AI semantic matching..."):
                        scored = score_jobs_against_resume(extracted, resume)
                        st.session_state["job_task_scored"] = scored
                    st.rerun()
        else:
            st.info("Upload your resume above to get match scores")

        # Display
        jobs_to_show = st.session_state.get("job_task_scored") or extracted
        show_match   = bool(
            st.session_state.get("job_task_scored") and
            resume and resume.get("skills")
        )

        if jobs_to_show:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Jobs", len(jobs_to_show))
            with m2:
                if show_match:
                    strong = sum(
                        1 for j in jobs_to_show
                        if j.get("match",{}).get("match_score",0) >= 60
                    )
                    st.metric("Good/Strong Matches", strong)
            with m3:
                min_score = (
                    st.slider("Min match %", 0, 100, 0, key="jt_min_s")
                    if show_match else 0
                )

            if show_match and min_score > 0:
                jobs_to_show = [
                    j for j in jobs_to_show
                    if j.get("match",{}).get("match_score",0) >= min_score
                ]

            if show_match:
                st.caption("⬆️ Sorted by match score — highest first")
            else:
                st.caption("Upload resume and click Score to get match percentages")

            for i, job in enumerate(jobs_to_show):
                _render_job_card(job, i, show_match=show_match)