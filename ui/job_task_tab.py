"""
Job Task Tab — Final.
Uses DOM-extracted job cards (not text-based).
Resume parsed section by section with correct experience calculation.
"""
import re
import json
import time
import streamlit as st
from functools import lru_cache
from utils.secure_logger import get_secure_logger
from utils.email_cleaner import parse_email_html, extract_job_cards, clean_email_body

logger = get_secure_logger(__name__)


# ── LLM helpers ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=500)
def _llm(prompt: str) -> str:
    from utils.llm_client import call_llm
    return call_llm(prompt, temperature=0.0, max_tokens=2000, use_cache=True)


def _llm_json(prompt: str, default):
    try:
        raw = _llm(prompt)
        m   = re.search(r'(\[.*\]|\{.*\})', raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
    except Exception as e:
        logger.warning(f"LLM JSON failed: {type(e).__name__}")
    return default


# ── PDF text extraction ─────────────────────────────────────────────────────────

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
    # Fix common PDF extraction issues (broken words)
    text = re.sub(r'(\w{3,})\s+(ously|tion|ing|ment|ness|ity|ance|ence)\b', r'\1\2', text)
    return text.strip()


# ── Section detection ───────────────────────────────────────────────────────────

_SECTION_RE = {
    "summary":        r'^(?:professional\s+)?(?:summary|objective|profile|about\s+me)\s*$',
    "education":      r'^(?:education|academic|qualification|educational\s+background)\s*$',
    "experience":     r'^(?:work\s+)?experience\s*$',
    "internship":     r'^internships?\s*$',
    "employment":     r'^(?:employment|work\s+history|professional\s+experience)\s*$',
    "skills":         r'^(?:technical\s+)?skills?\s*$|^(?:core\s+)?competencies\s*$|^technologies?\s*$',
    "projects":       r'^(?:projects?|project\s+work|personal\s+projects?|academic\s+projects?)\s*$',
    "certifications": r'^certifications?\s*$|^certificates?\s*$|^achievements?\s*$',
}


def _split_sections(text: str) -> dict[str,str]:
    lines    = text.split("\n")
    sections: dict[str,list] = {"header":[], "intro":[]}
    current  = "intro"

    # First few lines = header (name, contact)
    for i, line in enumerate(lines[:7]):
        stripped = line.strip()
        if stripped and len(stripped) < 80:
            sections["header"].append(stripped)
        if len(sections["header"]) >= 3:
            break

    start = len(sections["header"])
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            sections.setdefault(current,[]).append("")
            continue
        matched = None
        for sec_key, pattern in _SECTION_RE.items():
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 60:
                matched = sec_key
                break
        if matched:
            current = matched
            sections.setdefault(current,[])
        else:
            sections.setdefault(current,[]).append(line)

    return {k: "\n".join(v).strip() if isinstance(v,list) else v for k,v in sections.items()}


def _calc_experience_months(text: str) -> int:
    """Calculate months of experience from date ranges in text."""
    import datetime
    now_y = datetime.datetime.now().year
    now_m = datetime.datetime.now().month

    _MONTHS = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
        'january':1,'february':2,'march':3,'april':4,'june':6,'july':7,
        'august':8,'september':9,'october':10,'november':11,'december':12,
    }
    pattern = re.compile(
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
        r'january|february|march|april|june|july|august|september|october|november|december)'
        r'\.?\s+(\d{4})\s*[-–—to]+\s*'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
        r'january|february|march|april|june|july|august|september|october|november|december|'
        r'present|current|ongoing|now)'
        r'\.?\s*(\d{4})?',
        re.IGNORECASE
    )
    total = 0
    for m in pattern.finditer(text):
        try:
            s_m = _MONTHS.get(m.group(1).lower()[:3], 1)
            s_y = int(m.group(2))
            e_s = m.group(3).lower()[:3]
            if e_s in ('pre','cur','ong','now'):
                e_m, e_y = now_m, now_y
            else:
                e_m = _MONTHS.get(e_s, now_m)
                e_y = int(m.group(4)) if m.group(4) else now_y
            months = (e_y - s_y) * 12 + (e_m - s_m)
            if 0 < months < 600:
                total += months
        except Exception:
            pass

    if total == 0:
        yr = re.search(r'(\d+(?:\.\d+)?)\s*years?', text, re.IGNORECASE)
        mo = re.search(r'(\d+)\s*months?', text, re.IGNORECASE)
        if yr: total += int(float(yr.group(1)) * 12)
        if mo: total += int(mo.group(1))

    return total


# ── LLM prompts ─────────────────────────────────────────────────────────────────

_SKILLS_PROMPT = """\
Extract ALL technical skills from this section.
Return JSON array only. Min 3 chars each. Real technologies only.

{text}

["python","machine learning","sql"]"""

_EXP_PROMPT = """\
Extract work experience from this section only.
Return JSON:
{{"roles":[{{"title":"job title","company":"company","start":"Month Year","end":"Month Year or Present","bullets":["did X using Y"]}}],"summary":"2 sentences about this experience only"}}

{text}

JSON only:"""

_PROJ_PROMPT = """\
Extract projects.
Return JSON array:
[{{"name":"project name","description":"what it does","technologies":["tech1","tech2"]}}]

{text}

JSON array only:"""

_JOB_MATCH_PROMPT = """\
Score job-candidate match. Realistic score, not inflated.

JOB: {role} at {company} ({location})
Required skills: {job_skills}

CANDIDATE:
Skills: {resume_skills}
Experience: {exp_months} months
Current role: {current_role}
Education: {education}

Score 0-97 (NEVER 100):
Skill overlap: 50pts | Role fit: 30pts | Experience: 20pts

{{"match_score":65,"matched_skills":["python"],"missing_skills":["tableau"],"fit_reason":"Strong Python but lacks Tableau","strengths":"Python and ML","gaps":"Missing Tableau","recommendation":"Good Match","ready_to_apply":true,"skill_gap_suggestions":[{{"skill":"tableau","resource":"Tableau Public (free)"}}]}}

JSON only:"""


# ── Skill validation ─────────────────────────────────────────────────────────────

_VALID_SKILLS = {
    "python","java","javascript","typescript","c++","c#","golang","go","rust",
    "swift","kotlin","php","ruby","scala","r programming","matlab","bash",
    "flutter","react","reactjs","angular","vue","nextjs","nodejs","express",
    "django","flask","fastapi","spring boot","html","css","tailwind","bootstrap",
    "graphql","rest api","machine learning","deep learning","nlp","computer vision",
    "tensorflow","pytorch","keras","scikit-learn","sklearn","pandas","numpy",
    "scipy","matplotlib","seaborn","plotly","sql","mysql","postgresql","mongodb",
    "redis","elasticsearch","kafka","spark","hadoop","airflow","tableau","power bi",
    "excel","aws","azure","gcp","docker","kubernetes","ci/cd","git","linux",
    "devops","xgboost","lightgbm","transformers","bert","llm","openai","langchain",
    "rag","generative ai","genai","statistics","easyocr","opencv","streamlit",
    "model optimization","feature engineering","data analysis","data engineering",
    "microsoft excel","google analytics","a/b testing","time series",
}


def _validate_skills(skills: list) -> list[str]:
    if not skills: return []
    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str): continue
        s = s.strip().strip("\"'.,;:•").strip()
        if len(s) < 2 or len(s) > 60: continue
        if re.match(r'^[\d\s\W]+$', s): continue
        if re.match(r'^[a-zA-Z]$', s): continue
        s_lower = s.lower()
        is_valid = (
            s_lower in _VALID_SKILLS or
            (len(s) >= 3 and
             re.match(r'^[A-Za-z][A-Za-z0-9\s\+\#\-\.\/\_]+$', s) and
             not s_lower.startswith(("the ","and ","for ","with ","from ","this ")))
        )
        if is_valid and s_lower not in seen:
            seen.add(s_lower)
            cleaned.append(s)
    return cleaned[:25]


# ── Resume parsing ──────────────────────────────────────────────────────────────

def parse_resume(file_bytes: bytes, file_type: str = "pdf") -> dict:
    raw = (
        _extract_pdf_text(file_bytes)
        if file_type == "pdf"
        else file_bytes.decode("utf-8", errors="ignore")
    )
    raw = re.sub(r'\n{3,}','\n\n', raw).strip()
    if len(raw) < 100:
        return _empty_resume()

    sections = _split_sections(raw)
    result   = _empty_resume()
    result["raw_text_length"] = len(raw)

    # 1. Name — from header ONLY
    header = sections.get("header","")
    if isinstance(header, list):
        header = "\n".join(header)
    name = _llm(
        f"What is the full name in this resume header?\n"
        f"Return ONLY the complete full name (e.g. 'Ayush Kumar').\n\n{header[:200]}"
    ).strip()
    name = re.sub(r'[^A-Za-z\s\.\-]','',name).strip()
    if len(name) < 2 or len(name) > 60:
        for line in header.split("\n"):
            line = line.strip()
            if 3 < len(line) < 50 and re.search(r'[A-Z][a-z]',line) and not re.search(r'[@\d]',line):
                name = line
                break
    result["name"] = name

    # 2. Experience — from experience/internship sections ONLY
    exp_text = ""
    for key in ["experience","internship","internships","employment"]:
        content = sections.get(key,"")
        if content and len(content) > 20:
            exp_text += "\n" + content
    exp_text = exp_text.strip()

    if exp_text:
        exp_months = _calc_experience_months(exp_text)
        result["experience_years"] = round(exp_months / 12, 2)

        exp_data = _llm_json(_EXP_PROMPT.format(text=exp_text[:2000]), {})
        if isinstance(exp_data, dict):
            roles = exp_data.get("roles",[])
            if roles:
                result["roles"] = roles
                result["current_role"] = roles[0].get("title","")
                parts = []
                for role in roles[:3]:
                    t  = role.get("title","")
                    co = role.get("company","")
                    s  = role.get("start","")
                    e  = role.get("end","")
                    bl = role.get("bullets",[])
                    if t:
                        desc = f"{t} at {co} ({s}–{e})"
                        if bl:
                            desc += ": " + "; ".join(bl[:2])
                        parts.append(desc)
                result["experience"] = " | ".join(parts)
    else:
        result["experience_years"] = 0

    # 3. Skills — from skills + projects sections ONLY
    skills_sec  = sections.get("skills","") or sections.get("languages","") or ""
    proj_sec    = sections.get("projects","") or sections.get("project work","") or ""
    all_skills: list[str] = []

    if skills_sec and len(skills_sec) > 5:
        sk = _llm_json(_SKILLS_PROMPT.format(text=skills_sec[:1500]), [])
        if isinstance(sk, list): all_skills.extend(sk)

    if proj_sec and len(proj_sec) > 10:
        sk2 = _llm_json(
            f"Extract all technologies from these projects.\n"
            f"Return JSON array: [\"tech1\",\"tech2\"]\n\n{proj_sec[:1000]}",
            []
        )
        if isinstance(sk2, list): all_skills.extend(sk2)

        proj_data = _llm_json(_PROJ_PROMPT.format(text=proj_sec[:1500]), [])
        if isinstance(proj_data, list):
            result["projects"] = [
                f"{p.get('name','?')}: {p.get('description','')[:80]} [{', '.join(p.get('technologies',[])[:4])}]"
                for p in proj_data[:5] if p.get("name")
            ]

    result["skills"] = _validate_skills(all_skills)

    # 4. Education
    edu_sec = sections.get("education","") or sections.get("academic","") or ""
    if edu_sec:
        edu = _llm(
            f"Extract education as: Degree, Field, Institution, Year\n"
            f"Example: B.Tech., Computer Science, VIT Vellore, 2026\n\n"
            f"{edu_sec[:500]}\nReturn one line only:"
        ).strip()
        result["education"] = edu[:200]

    # 5. Certifications
    cert_sec = sections.get("certifications","") or ""
    if cert_sec:
        certs = _llm_json(f"List certifications as JSON array:\n{cert_sec[:500]}\nReturn: [\"cert1\"]", [])
        if isinstance(certs, list):
            result["certifications"] = [str(c) for c in certs[:5]]

    # 6. Summary
    exp_m = int(result["experience_years"] * 12)
    ctx   = f"Name: {result['name']}\nRole: {result['current_role']}\nExperience: {exp_m} months\n"
    ctx  += f"Skills: {', '.join(result['skills'][:8])}\nEducation: {result['education']}"
    result["summary"] = _llm(
        f"Write a 2-sentence professional summary for:\n{ctx}\n"
        f"Focus on technical strengths and career stage.\nReturn 2 sentences only:"
    ).strip()

    return result


def _empty_resume() -> dict:
    return {
        "name":"","skills":[],"experience_years":0,"experience":"",
        "current_role":"","education":"","projects":[],"certifications":[],
        "summary":"","raw_text_length":0,"roles":[],
    }


# ── Job extraction ──────────────────────────────────────────────────────────────

def _get_job_emails(source: str) -> list[dict]:
    """Get job-related emails."""
    from memory.repository import get_all_emails, get_all_processed
    all_emails = get_all_emails(source=source)
    proc_list  = get_all_processed(source=source)
    proc_map   = {p["email_id"]: p for p in proc_list}

    _JOB_SENDERS = [
        "glassdoor","naukri","linkedin","indeed","monster",
        "resume.io","jobs@","career@","hiring@","recruit",
    ]
    _JOB_CATS = {"Job / Recruitment","Job","job_recruitment"}

    job_emails = []
    for email in all_emails:
        cat    = proc_map.get(email.get("id",""),{}).get("category","")
        sender = email.get("sender","").lower()
        if cat in _JOB_CATS or any(js in sender for js in _JOB_SENDERS):
            job_emails.append(email)
    return job_emails


def extract_jobs_from_emails(emails: list[dict]) -> list[dict]:
    """
    Extract jobs using DOM structure-aware parsing.
    Uses extract_job_cards() which does DOM traversal, not text parsing.
    """
    all_jobs = []

    for email in emails:
        body = email.get("body","")
        if not body:
            continue

        # Primary: DOM structure extraction
        cards = extract_job_cards(body)

        if cards:
            for card in cards:
                card["email_id"]      = email.get("id","")
                card["email_subject"] = email.get("subject","")
                card["source"]        = "dom_structure"
                if "all_links" not in card:
                    card["all_links"] = [card["link"]] if card.get("link") else []
            all_jobs.extend(cards)
            logger.info(f"DOM found {len(cards)} jobs from {email.get('subject','')[:40]}")
        else:
            # LLM fallback
            time.sleep(0.2)
            clean = clean_email_body(body, max_chars=3500)
            if not clean or len(clean) < 50:
                continue

            prompt = (
                f"This is a job alert email. Extract ALL job listings.\n"
                f"Subject: {email.get('subject','')}\n\n"
                f"Content:\n{clean[:3000]}\n\n"
                f"Return JSON array (typically 5-12 jobs):\n"
                f'[{{"role":"title","company":"co","location":"city","salary":"range or Not specified","skills":["skill1"]}}]\n'
                f"JSON only:"
            )
            jobs_data = _llm_json(prompt, [])

            parsed    = parse_email_html(body)
            job_links = parsed.get("job_links",[])
            all_links = parsed.get("links",[])
            links     = job_links if job_links else all_links

            if isinstance(jobs_data, list):
                for i, job in enumerate(jobs_data):
                    if not isinstance(job, dict): continue
                    role = str(job.get("role","")).strip()
                    if len(role) < 3: continue
                    link = links[i] if i < len(links) else None
                    all_jobs.append({
                        "role":          role[:100],
                        "company":       str(job.get("company","Unknown"))[:100],
                        "location":      str(job.get("location","Not specified"))[:80],
                        "salary":        str(job.get("salary","Not specified"))[:100],
                        "skills":        _validate_skills(job.get("skills",[])),
                        "link":          link,
                        "all_links":     [link] if link else [],
                        "email_id":      email.get("id",""),
                        "email_subject": email.get("subject",""),
                        "source":        "llm_fallback",
                    })
            logger.info(f"LLM found {len(jobs_data) if isinstance(jobs_data,list) else 0} jobs from {email.get('subject','')[:40]}")

    # Deduplicate
    seen, unique = set(), []
    for job in all_jobs:
        key = (
            re.sub(r'\s+','',job.get("role","")).lower()[:50],
            re.sub(r'\s+','',job.get("company","")).lower()[:30],
        )
        if key not in seen and key[0] and key[0] != "unknown":
            seen.add(key)
            unique.append(job)

    logger.info(f"Total unique jobs extracted: {len(unique)}")
    return unique


# ── Job scoring ─────────────────────────────────────────────────────────────────

def score_jobs(jobs: list[dict], resume: dict) -> list[dict]:
    resume_skills = set(s.lower() for s in _validate_skills(resume.get("skills",[])))
    scored = []
    exp_months = int(resume.get("experience_years",0) * 12)

    for job in jobs:
        job_skills = set(s.lower() for s in _validate_skills(job.get("skills",[])))
        matched    = sorted(job_skills & resume_skills)
        missing    = sorted(job_skills - resume_skills)

        if job_skills:
            overlap    = (len(matched) / len(job_skills)) * 50
            rule_score = min(97, int(overlap + 20))
        else:
            rule_score = 30

        def _rec(s):
            if s >= 80: return "Strong Match"
            if s >= 60: return "Good Match"
            if s >= 40: return "Partial Match"
            return "Weak Match"

        try:
            prompt = _JOB_MATCH_PROMPT.format(
                role=job.get("role",""),
                company=job.get("company",""),
                location=job.get("location",""),
                job_skills=", ".join(list(job_skills)[:12]),
                resume_skills=", ".join(list(resume_skills)[:18]),
                exp_months=exp_months,
                current_role=resume.get("current_role",""),
                education=resume.get("education","")[:80],
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
                    "ready_to_apply":        bool(data.get("ready_to_apply",score>=50)),
                    "skill_gap_suggestions": data.get("skill_gap_suggestions",[]) or [],
                }
            else:
                raise ValueError
        except Exception:
            match = {
                "match_score":           rule_score,
                "matched_skills":        list(matched),
                "missing_skills":        list(missing),
                "fit_reason":            f"{len(matched)}/{len(job_skills)} skills matched",
                "strengths":             f"Matched: {', '.join(list(matched)[:3])}",
                "gaps":                  f"Missing: {', '.join(list(missing)[:3])}",
                "recommendation":        _rec(rule_score),
                "ready_to_apply":        rule_score >= 50,
                "skill_gap_suggestions": [{"skill":s,"resource":f"Search '{s}' on Coursera"} for s in list(missing)[:2]],
            }

        scored.append({**job, "match": match})

    scored.sort(key=lambda x: x.get("match",{}).get("match_score",0), reverse=True)
    return scored


# ── UI ──────────────────────────────────────────────────────────────────────────

_REC_COLORS = {
    "Strong Match":"#22c55e","Good Match":"#84cc16",
    "Partial Match":"#f59e0b","Weak Match":"#ef4444",
}


def _render_job_card(job: dict, idx: int, show_match: bool):
    match     = job.get("match",{})
    score     = match.get("match_score",0) if show_match else None
    rec       = match.get("recommendation","") if show_match else ""
    rec_color = _REC_COLORS.get(rec,"#6b7280")
    ready     = match.get("ready_to_apply",False) if show_match else False

    with st.container():
        ct, cs = st.columns([4,1])
        with ct:
            st.markdown(f"#### 💼 {job.get('role','?')}")
            parts = []
            co = job.get("company","")
            lo = job.get("location","")
            sa = job.get("salary","")
            if co and co not in ("Unknown",""): parts.append(f"🏢 **{co}**")
            if lo and lo != "Not specified":    parts.append(f"📍 {lo}")
            if sa and sa != "Not specified":    parts.append(f"💰 {sa}")
            if parts: st.markdown(" &nbsp;|&nbsp; ".join(parts))
        with cs:
            if show_match and score is not None:
                st.markdown(
                    f'<div style="background:{rec_color};color:white;border-radius:8px;'
                    f'padding:8px;text-align:center;font-weight:bold;font-size:22px;">{score}%</div>'
                    f'<div style="text-align:center;font-size:11px;color:{rec_color};'
                    f'font-weight:bold;margin-top:4px">{rec}</div>',
                    unsafe_allow_html=True,
                )
                if ready: st.success("✅ Apply!")

        if job.get("skills"):
            tags = " ".join(
                f'<span style="background:#e2e8f0;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;">{s}</span>'
                for s in job["skills"][:12]
            )
            st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)

        if show_match and match:
            c1,c2 = st.columns(2)
            with c1:
                if match.get("matched_skills"):
                    st.markdown("✅ " + ", ".join(f"`{s}`" for s in match["matched_skills"][:5]))
                if match.get("fit_reason"):
                    st.caption(f"💡 {match['fit_reason']}")
            with c2:
                if match.get("missing_skills"):
                    st.markdown("⚠️ " + ", ".join(f"`{s}`" for s in match["missing_skills"][:5]))
                if match.get("gaps"):
                    st.caption(f"📌 {match['gaps']}")
            sgs = match.get("skill_gap_suggestions",[])
            if sgs:
                with st.expander("📚 Learning Path"):
                    for sg in sgs[:3]:
                        st.write(f"• **{sg.get('skill','')}** — {sg.get('resource','')}")

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
        src_label = {"dom_structure":"🔗 Extracted from HTML","llm_fallback":"🤖 AI Extracted"}.get(job.get("source",""),"📧 Extracted")
        st.caption(f"{src_label} | From: {truncate(job.get('email_subject',''),50)}")
        st.markdown("---")


def truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


def render_job_task_tab():
    st.header("💼 Job Task — Resume & Job Matching")

    # Step 1: Resume
    st.subheader("📄 Step 1: Upload Your Resume")
    st.caption("Name ← header | Experience ← work/internship sections | Skills ← skills/projects sections")

    cu, cs = st.columns([2,2])
    with cu:
        uploaded = st.file_uploader("PDF or TXT", type=["pdf","txt"], key="jt_upload")

    resume = st.session_state.get("jt_resume")
    if uploaded:
        if st.session_state.get("jt_fname") != uploaded.name or not resume:
            with st.spinner("Parsing resume section by section..."):
                fb     = uploaded.read()
                ft     = "pdf" if uploaded.name.lower().endswith(".pdf") else "txt"
                resume = parse_resume(fb, ft)
                st.session_state["jt_resume"] = resume
                st.session_state["jt_fname"]  = uploaded.name

    with cs:
        if resume and resume.get("name"):
            st.success(f"✅ **{resume['name']}**")
            exp_m = int(resume.get("experience_years",0) * 12)
            st.caption(f"Skills: {len(resume.get('skills',[]))} | Exp: {exp_m} months | Role: {resume.get('current_role','—')}")
        elif resume:
            st.warning("Resume parsed — some fields may be incomplete")
        else:
            st.info("📤 Upload resume to get match scores")

    if resume and (resume.get("name") or resume.get("skills")):
        with st.expander("📋 Parsed Resume Details"):
            c1,c2 = st.columns(2)
            with c1:
                st.write(f"**Name:** {resume.get('name','—')}")
                exp_m = int(resume.get("experience_years",0) * 12)
                st.write(f"**Experience:** {exp_m} months")
                st.write(f"**Current Role:** {resume.get('current_role','—')}")
                st.write(f"**Education:** {resume.get('education','—')}")
                if resume.get("experience"):
                    st.write("**Work History:**")
                    st.caption(resume["experience"][:400])
            with c2:
                if resume.get("skills"):
                    tags = " ".join(
                        f'<span style="background:#dcfce7;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px;">{s}</span>'
                        for s in resume["skills"][:20]
                    )
                    st.markdown(f"**Skills ({len(resume['skills'])}):**")
                    st.markdown(tags, unsafe_allow_html=True)
                if resume.get("projects"):
                    st.write("**Projects:**")
                    for p in resume["projects"][:4]:
                        st.caption(f"• {str(p)[:120]}")
                if resume.get("summary"):
                    st.write("**Summary:**")
                    st.caption(resume["summary"])

    st.divider()

    # Step 2: Extract Jobs
    st.subheader("🔍 Step 2: Extract Jobs from Emails")
    source     = "gmail" if st.session_state.get("authenticated") else "demo"
    job_emails = _get_job_emails(source)

    if not job_emails:
        st.info(f"No job emails found. Process emails in the **📥 Inbox** tab first.")
        return

    st.success(f"Found **{len(job_emails)} job emails** — each typically has 5-12 listings")

    c1,c2,c3 = st.columns(3)
    with c1:
        max_e = st.slider("Emails to process", 1, min(len(job_emails),20), min(len(job_emails),5), key="jt_max")
    with c2:
        st.write("")
        ext_btn = st.button("🚀 Extract All Jobs", type="primary", use_container_width=True, key="jt_ext")
    with c3:
        st.write("")
        if st.button("🗑️ Clear", use_container_width=True, key="jt_clr"):
            st.session_state.pop("jt_jobs",None)
            st.session_state.pop("jt_scored",None)
            st.rerun()

    if ext_btn:
        to_proc = job_emails[:max_e]
        prog    = st.progress(0)
        status  = st.empty()
        status.text(f"Extracting jobs from {len(to_proc)} emails...")
        jobs = extract_jobs_from_emails(to_proc)
        prog.progress(1.0)
        st.session_state["jt_jobs"]   = jobs
        st.session_state.pop("jt_scored",None)
        status.success(f"✅ Found **{len(jobs)} unique jobs** from {len(to_proc)} emails!")

    st.divider()

    # Step 3: Display and score
    extracted = st.session_state.get("jt_jobs",[])
    if not extracted:
        st.info("Click **🚀 Extract All Jobs** above to find job listings.")
        return

    st.subheader(f"📊 Step 3: {len(extracted)} Jobs Found")

    if resume and resume.get("skills") and not st.session_state.get("jt_scored"):
        if st.button(f"🎯 Score {len(extracted)} Jobs Against Resume", type="secondary", key="jt_score"):
            with st.spinner("AI scoring..."):
                scored = score_jobs(extracted, resume)
                st.session_state["jt_scored"] = scored
            st.rerun()

    jobs_to_show = st.session_state.get("jt_scored") or extracted
    show_match   = bool(st.session_state.get("jt_scored") and resume and resume.get("skills"))

    m1,m2,m3 = st.columns(3)
    with m1: st.metric("Total Jobs", len(jobs_to_show))
    with m2:
        if show_match:
            good = sum(1 for j in jobs_to_show if j.get("match",{}).get("match_score",0)>=60)
            st.metric("Good/Strong Matches", good)
    with m3:
        min_s = st.slider("Min match %",0,100,0,key="jt_mins") if show_match else 0

    if show_match and min_s > 0:
        jobs_to_show = [j for j in jobs_to_show if j.get("match",{}).get("match_score",0)>=min_s]

    if show_match:
        st.caption("⬆️ Sorted highest match first")
    else:
        st.caption("Upload resume and click Score to get match percentages")

    for i, job in enumerate(jobs_to_show):
        _render_job_card(job, i, show_match)