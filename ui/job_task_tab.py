"""
Job Task Tab — Final production version.

Fixes:
1. Experience read ONLY from experience/internship section (not summary)
2. Experience years calculated from dates only (3 months = 0.25 years)
3. Skills read ONLY from skills + projects sections
4. Name from header/first line only
5. Glassdoor email job extraction: finds job blocks by bold title + Easy Apply pattern
6. LLM extraction as powerful fallback
7. Semantic matching with full candidate context
"""
import re
import json
import time
import streamlit as st
from functools import lru_cache
from utils.secure_logger import get_secure_logger
from utils.email_cleaner import clean_email_body, parse_email_html

logger = get_secure_logger(__name__)


# ── Cached LLM ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=500)
def _llm(prompt: str) -> str:
    from utils.llm_client import call_llm
    return call_llm(prompt, temperature=0.0, max_tokens=2000, use_cache=True)


def _llm_json(prompt: str, default):
    try:
        raw = _llm(prompt)
        # Find JSON in response
        m = re.search(r'(\[.*\]|\{.*\})', raw, re.DOTALL)
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
    # Fix common PDF broken words
    text = re.sub(r'(\w{3,})\s+(ously|tion|ing|ment|ness|ity|ance|ence)\b', r'\1\2', text)
    return text.strip()


# ── Section detection ───────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "header":         None,  # First few lines
    "summary":        r'^(?:professional\s+)?(?:summary|objective|profile|about\s+me)\s*$',
    "education":      r'^(?:education|academic|qualification|educational\s+background)\s*$',
    "experience":     r'^(?:work\s+)?experience\s*$',
    "internship":     r'^internships?\s*$',
    "employment":     r'^(?:employment|work\s+history|professional\s+experience)\s*$',
    "skills":         r'^(?:technical\s+)?skills?\s*$|^(?:core\s+)?competencies\s*$|^technologies?\s*$',
    "projects":       r'^(?:projects?|project\s+work|personal\s+projects?|academic\s+projects?)\s*$',
    "certifications": r'^certifications?\s*$|^certificates?\s*$|^achievements?\s*$',
    "languages":      r'^(?:programming\s+)?languages?\s*$',
}


def _split_sections(text: str) -> dict[str, str]:
    """
    Split resume into sections by detecting headers.
    Returns {section_name: content_text}
    """
    lines   = text.split("\n")
    sections: dict[str, list[str]] = {"header": [], "intro": []}
    current = "intro"

    # First 5 lines → header (name, contact info)
    header_lines = []
    content_start = 0
    for i, line in enumerate(lines[:8]):
        stripped = line.strip()
        if stripped and len(stripped) < 80:
            header_lines.append(stripped)
            content_start = i + 1
        if len(header_lines) >= 3:
            break
    sections["header"] = header_lines

    # Rest → detect sections
    for line in lines[content_start:]:
        stripped = line.strip()
        if not stripped:
            if current in sections:
                sections[current].append("")
            continue

        # Check if this line is a section header
        matched = None
        for sec_key, pattern in _SECTION_PATTERNS.items():
            if sec_key == "header" or not pattern:
                continue
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 60:
                matched = sec_key
                break

        if matched:
            current = matched
            if current not in sections:
                sections[current] = []
        else:
            if current not in sections:
                sections[current] = []
            sections[current].append(line)

    # Convert lists to strings
    result = {}
    for key, val in sections.items():
        if isinstance(val, list):
            result[key] = "\n".join(val).strip()
        else:
            result[key] = "\n".join(val).strip() if isinstance(val, list) else str(val)

    return result


# ── Date/duration extraction ────────────────────────────────────────────────────

def _calculate_experience_months(text: str) -> int:
    """
    Calculate total experience in months from date ranges in text.
    Handles: May 2025 – August 2025, Jan 2024 - Present, etc.
    Returns months as integer.
    """
    months_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
        'january':1,'february':2,'march':3,'april':4,'june':6,'july':7,
        'august':8,'september':9,'october':10,'november':11,'december':12,
    }

    import datetime
    current_year  = datetime.datetime.now().year
    current_month = datetime.datetime.now().month
    total_months  = 0

    # Pattern: Month Year – Month Year or Month Year - Present
    date_range_pattern = re.compile(
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|'
        r'march|april|june|july|august|september|october|november|december)'
        r'\.?\s+(\d{4})\s*[-–—]\s*'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|'
        r'march|april|june|july|august|september|october|november|december|present|current|ongoing)'
        r'\.?\s*(\d{4})?',
        re.IGNORECASE
    )

    for m in date_range_pattern.finditer(text):
        try:
            start_m = months_map.get(m.group(1).lower()[:3], 1)
            start_y = int(m.group(2))
            end_str = m.group(3).lower()[:3]

            if end_str in ('pre','cur','ong'):
                end_m = current_month
                end_y = current_year
            else:
                end_m = months_map.get(end_str, current_month)
                end_y = int(m.group(4)) if m.group(4) else current_year

            months = (end_y - start_y) * 12 + (end_m - start_m)
            if 0 < months < 600:  # sanity check
                total_months += months
        except Exception:
            pass

    # Also look for "X months" or "X years" text
    if total_months == 0:
        yr_match = re.search(r'(\d+(?:\.\d+)?)\s*years?', text, re.IGNORECASE)
        mo_match = re.search(r'(\d+)\s*months?', text, re.IGNORECASE)
        if yr_match:
            total_months += int(float(yr_match.group(1)) * 12)
        if mo_match:
            total_months += int(mo_match.group(1))

    return total_months


# ── LLM prompts ─────────────────────────────────────────────────────────────────

_NAME_PROMPT = """\
What is the full name of the person in this resume header?
Return ONLY the full name (e.g. "Ayush Kumar"). Nothing else.
If unclear, return "Unknown".

Header:
{header}"""

_SKILLS_PROMPT = """\
List all technical skills from this section.
Return JSON array of skill strings only.
Rules: real technologies only, min 3 chars, no single letters.

Text:
{text}

Return: ["python", "machine learning", "sql"]"""

_EXPERIENCE_PROMPT = """\
Extract work experience details from this section ONLY.

Section text:
{text}

Return JSON:
{{
  "roles": [
    {{
      "title": "job title",
      "company": "company name",
      "start": "Month Year",
      "end": "Month Year or Present",
      "description": "2-3 bullet points of what was done"
    }}
  ],
  "summary": "2-3 sentences describing ONLY this work experience"
}}

IMPORTANT: Only describe what's written in this section. Do not infer or add details.
Return JSON only:"""

_PROJECT_PROMPT = """\
Extract projects from this section.

Section:
{text}

Return JSON array:
[
  {{
    "name": "project name",
    "description": "what it does and what problem it solves",
    "technologies": ["tech1", "tech2", "tech3"]
  }}
]

Be specific about what each project actually does.
Return JSON array only:"""

_JOB_EXTRACT_PROMPT = """\
This is a job alert email from Glassdoor/Naukri. Extract ALL job listings.

Subject: {subject}

Email text content:
{body}

Each job in Glassdoor emails has this structure:
- Job title (bold)
- Company name with rating
- Location (India/Remote/City)
- Salary range (if available)
- Skills/tags
- "Easy Apply" button

Extract EVERY job. Return JSON array:
[
  {{
    "role": "Data Analyst",
    "company": "Terrier Security Services",
    "location": "India",
    "salary": "20K - 42K",
    "skills": ["statistics","microsoft excel","data analysis"],
    "description": "Data analyst role at Terrier Security Services"
  }}
]

Find ALL jobs in the email — there are typically 5-12.
Return [] only if truly no jobs.
JSON array only:"""

_MATCH_PROMPT = """\
Score how well this job fits this candidate. Be specific and realistic.

JOB:
Title: {role}
Company: {company}
Location: {location}
Skills needed: {job_skills}

CANDIDATE:
Name: {name}
Skills: {resume_skills}
Experience: {exp_months} months ({exp_years} years)
Current/Last Role: {current_role}
Education: {education}
Projects: {projects}

Score 0-97 (NEVER 100 unless absolutely every requirement matched):
- Skill match: 50 points max (how many required skills does candidate have)
- Role fit: 30 points max (how similar is the role to candidate's background)  
- Experience fit: 20 points max (is experience level appropriate)

{{
  "match_score": 72,
  "matched_skills": ["python", "sql"],
  "missing_skills": ["tableau"],
  "fit_reason": "Strong Python and ML skills align well with Data Analyst role at {company}. 3 months internship experience is entry level appropriate.",
  "strengths": "Python expertise, ML projects, relevant internship",
  "gaps": "Needs Tableau, limited professional experience",
  "recommendation": "Good Match",
  "ready_to_apply": true,
  "skill_gap_suggestions": [
    {{"skill": "tableau", "resource": "Tableau Public (free) or Coursera Tableau course"}}
  ]
}}

recommendation: "Strong Match" (80+), "Good Match" (60-79), "Partial Match" (40-59), "Weak Match" (<40)
JSON only:"""


# ── Skill validation ────────────────────────────────────────────────────────────

_VALID_SKILLS_SET = {
    "python","java","javascript","typescript","c++","c#","golang","go","rust",
    "swift","kotlin","php","ruby","scala","r programming","matlab","bash",
    "shell","flutter","react","reactjs","angular","vue","nextjs","nodejs",
    "express","django","flask","fastapi","spring","spring boot","html","css",
    "tailwind","bootstrap","graphql","rest api","machine learning","deep learning",
    "nlp","natural language processing","computer vision","tensorflow","pytorch",
    "keras","scikit-learn","sklearn","pandas","numpy","scipy","matplotlib",
    "seaborn","plotly","sql","mysql","postgresql","sqlite","mongodb","redis",
    "elasticsearch","kafka","spark","hadoop","airflow","dbt","tableau","power bi",
    "excel","data analysis","data engineering","etl","mlops","llm","openai",
    "langchain","rag","vector database","embeddings","prompt engineering",
    "aws","azure","gcp","docker","kubernetes","ci/cd","jenkins","github actions",
    "terraform","linux","git","devops","microservices","system design","agile",
    "xgboost","lightgbm","transformers","bert","hugging face","generative ai",
    "statistics","probability","feature engineering","model deployment",
    "reinforcement learning","time series","a/b testing","data visualization",
    "easyocr","ocr","opencv","pillow","streamlit","fastapi","flask","django",
    "genai","rag pipeline","semantic search","vector search","chromadb","pinecone",
    "model optimization","hyperparameter tuning","cross validation",
    "microsoft excel","microsoft office","google analytics",
}


def _validate_skills(skills: list) -> list[str]:
    if not skills:
        return []
    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s = s.strip().strip("\"'.,;:•-").strip()
        if len(s) < 2 or len(s) > 60:
            continue
        if re.match(r'^[\d\s\W]+$', s):
            continue
        if re.match(r'^[a-zA-Z]$', s):
            continue
        if re.search(r'[{};:@#<>]', s):
            continue
        s_lower = s.lower()
        # Accept if in whitelist OR looks like real tech (3+ chars, alphanumeric)
        is_valid = (
            s_lower in _VALID_SKILLS_SET or
            (len(s) >= 3 and
             re.match(r'^[A-Za-z][A-Za-z0-9\s\+\#\-\.\/\_]+$', s) and
             not s_lower.startswith(("the ", "and ", "for ", "with ", "from ", "this ")))
        )
        if is_valid and s_lower not in seen:
            seen.add(s_lower)
            cleaned.append(s)
    return cleaned[:25]


# ── Resume parsing ──────────────────────────────────────────────────────────────

def parse_resume(file_bytes: bytes, file_type: str = "pdf") -> dict:
    """
    Parse resume using section detection.
    - Name: from header section ONLY
    - Experience: from experience/internship section ONLY, months calculated from dates
    - Skills: from skills + projects sections ONLY
    """
    # Extract raw text
    if file_type == "pdf":
        raw = _extract_pdf_text(file_bytes)
    else:
        raw = file_bytes.decode("utf-8", errors="ignore")
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()

    if len(raw.strip()) < 100:
        return _empty_resume()

    # Split into sections
    sections = _split_sections(raw)

    result = _empty_resume()
    result["raw_text_length"] = len(raw)

    # ── 1. Name from header ONLY ───────────────────────────────────────────────
    header_text = sections.get("header","")
    if isinstance(header_text, list):
        header_text = "\n".join(header_text)

    name = _llm(
        f"What is the full name in this resume header?\n"
        f"Return ONLY the full name, nothing else.\n\n{header_text[:200]}"
    ).strip()
    # Clean name: remove non-name characters
    name = re.sub(r'[^A-Za-z\s\.\-]', '', name).strip()
    # Validate: should look like a real name (2+ words or proper case)
    if len(name) < 2 or len(name) > 60 or name.lower() in ("unknown","none","n/a"):
        # Fallback: first line that looks like a name
        for line in (header_text if isinstance(header_text, str) else "").split("\n"):
            line = line.strip()
            if (3 < len(line) < 50 and
                not re.search(r'[@\d\|/\\+]', line) and
                re.search(r'[A-Z][a-z]', line)):
                name = line
                break
    result["name"] = name

    # ── 2. Experience from experience/internship section ONLY ──────────────────
    exp_section = ""
    for key in ["experience", "internship", "internships", "employment"]:
        content = sections.get(key, "")
        if content and len(content) > 20:
            exp_section += "\n" + content

    if exp_section.strip():
        # Calculate months from dates in this section ONLY
        exp_months = _calculate_experience_months(exp_section)
        result["experience_years"] = round(exp_months / 12, 1)

        # LLM for structured extraction
        exp_data = _llm_json(
            _EXPERIENCE_PROMPT.format(text=exp_section[:2000]),
            {}
        )
        if isinstance(exp_data, dict):
            roles = exp_data.get("roles", [])
            if roles:
                result["roles"] = roles
                # Current role = most recent
                result["current_role"] = roles[0].get("title", "") if roles else ""
                # Build experience summary
                summaries = []
                for role in roles[:3]:
                    title   = role.get("title", "")
                    company = role.get("company", "")
                    start   = role.get("start", "")
                    end     = role.get("end", "")
                    desc    = role.get("description", "")
                    if title and company:
                        summaries.append(f"{title} at {company} ({start}–{end}): {desc}")
                result["experience"] = " | ".join(summaries)
            elif exp_data.get("summary"):
                result["experience"] = exp_data["summary"]
    else:
        result["experience_years"] = 0
        result["experience"]       = ""

    # ── 3. Skills from skills + projects sections ONLY ────────────────────────
    skills_section   = sections.get("skills","") or sections.get("languages","") or ""
    projects_section = sections.get("projects","") or sections.get("project work","") or ""

    all_skills: list[str] = []

    if skills_section and len(skills_section) > 5:
        sk = _llm_json(
            _SKILLS_PROMPT.format(text=skills_section[:1500]),
            []
        )
        if isinstance(sk, list):
            all_skills.extend(sk)

    if projects_section and len(projects_section) > 10:
        # Extract technologies from projects
        proj_sk = _llm_json(
            f"Extract all technologies/skills from these projects.\n"
            f"Return JSON array of technology names: [\"tech1\", \"tech2\"]\n\n"
            f"{projects_section[:1000]}",
            []
        )
        if isinstance(proj_sk, list):
            all_skills.extend(proj_sk)

        # Parse project details
        proj_data = _llm_json(
            _PROJECT_PROMPT.format(text=projects_section[:1500]),
            []
        )
        if isinstance(proj_data, list):
            result["projects"] = [
                f"{p.get('name','?')}: {p.get('description','')[:100]} "
                f"[{', '.join(p.get('technologies',[])[:4])}]"
                for p in proj_data[:5]
                if p.get("name")
            ]

    result["skills"] = _validate_skills(all_skills)

    # ── 4. Education ────────────────────────────────────────────────────────────
    edu_section = sections.get("education","") or sections.get("academic","") or ""
    if edu_section:
        edu = _llm(
            f"Extract education as: Degree, Field, Institution, Year\n"
            f"Example: B.Tech., Computer Science, VIT Vellore, 2026\n\n"
            f"{edu_section[:500]}\n\nReturn one line only:"
        ).strip()
        result["education"] = edu[:200]

    # ── 5. Certifications ───────────────────────────────────────────────────────
    cert_section = sections.get("certifications","") or sections.get("achievements","") or ""
    if cert_section:
        certs = _llm_json(
            f"List certifications as JSON array:\n{cert_section[:500]}\n"
            f'Return: ["cert1", "cert2"]',
            []
        )
        if isinstance(certs, list):
            result["certifications"] = [str(c) for c in certs[:5]]

    # ── 6. Professional summary ─────────────────────────────────────────────────
    summary_parts = []
    if result["name"]:
        summary_parts.append(f"Name: {result['name']}")
    if result["current_role"]:
        summary_parts.append(f"Role: {result['current_role']}")
    if result["experience_years"] > 0:
        months = int(result["experience_years"] * 12)
        summary_parts.append(f"Experience: {months} months")
    if result["skills"]:
        summary_parts.append(f"Skills: {', '.join(result['skills'][:8])}")
    if result["education"]:
        summary_parts.append(f"Education: {result['education'][:80]}")

    if summary_parts:
        summary = _llm(
            f"Write a 2-sentence professional summary for this candidate:\n"
            f"{chr(10).join(summary_parts)}\n\n"
            f"Focus on their technical strengths and career stage.\n"
            f"Return only the 2-sentence summary:"
        ).strip()
        result["summary"] = summary

    return result


def _empty_resume() -> dict:
    return {
        "name": "", "skills": [], "experience_years": 0, "experience": "",
        "current_role": "", "education": "", "projects": [],
        "certifications": [], "summary": "", "raw_text_length": 0, "roles": [],
    }


# ── Glassdoor-specific job extraction ─────────────────────────────────────────

def _extract_glassdoor_jobs_from_html(html: str, email: dict) -> list[dict]:
    """
    Extract jobs from Glassdoor email HTML.
    Glassdoor structure: bold title → company → location → salary → tags → Easy Apply link
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup   = BeautifulSoup(html, "html.parser")
    jobs   = []
    seen   = set()

    # Get all apply links (these are the job application URLs)
    apply_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if (href.startswith(("http://","https://")) and
            ("glassdoor.co.in" in href or "glassdoor.com" in href or
             "naukri.com" in href or "apply" in text or "easy apply" in text)):
            apply_links.append(href)

    # Find job title elements (bold tags or table cells with job title patterns)
    job_title_re = re.compile(
        r'\b(engineer|developer|analyst|scientist|manager|intern|designer|'
        r'architect|lead|senior|junior|associate|consultant|specialist|'
        r'director|executive|coordinator|fullstack|full.stack|frontend|'
        r'backend|devops|sde|swe|ml|ai|data|software|machine learning|'
        r'generative|applied)\b',
        re.IGNORECASE
    )

    # Strategy: find all bold/strong elements with job title text
    bold_elements = soup.find_all(["b", "strong"])
    link_idx      = 0

    for bold in bold_elements:
        title = bold.get_text(strip=True)
        if not (5 <= len(title) <= 120):
            continue
        if not job_title_re.search(title):
            continue
        title_lower = title.lower()
        if title_lower in seen:
            continue
        seen.add(title_lower)

        # Get context: look at parent and siblings for company/location/salary
        context_text = ""
        parent = bold.parent
        for _ in range(5):
            if parent:
                context_text = parent.get_text(separator=" ", strip=True)
                if len(context_text) > 30:
                    break
                parent = parent.parent

        # Extract details from context
        company  = _extract_company_from_context(context_text, title)
        location = _extract_location(context_text)
        salary   = _extract_salary(context_text)
        skills   = _extract_skills_from_text(context_text)

        # Assign apply link
        link = apply_links[link_idx] if link_idx < len(apply_links) else None
        link_idx += 1

        jobs.append({
            "role":          title[:100],
            "company":       company[:100],
            "location":      location,
            "salary":        salary,
            "skills":        skills,
            "link":          link,
            "all_links":     [link] if link else [],
            "description":   f"{title} at {company} — {location}",
            "email_id":      email.get("id",""),
            "email_subject": email.get("subject",""),
            "source":        "glassdoor_bs4",
        })

    return jobs


def _extract_company_from_context(context: str, role: str) -> str:
    """Extract company name from job context text."""
    # Remove the role title to avoid confusion
    context_clean = context.replace(role, "").strip()

    # Common patterns after job title
    patterns = [
        r'\b([A-Z][A-Za-z\s&\.]{3,40}?)\s+\d+\.\d+\s*(?:stars?)?',  # Company 4.5 stars
        r'\b([A-Z][A-Za-z\s&\.]{3,40}?)\s+(?:India|Remote|Bangalore|Mumbai|Delhi)',
    ]
    for pat in patterns:
        m = re.search(pat, context_clean)
        if m:
            co = m.group(1).strip()
            if co and len(co) > 2:
                return co

    # Fallback: second line of context
    lines = [l.strip() for l in context_clean.split("\n") if l.strip()]
    for line in lines[1:4]:
        if (3 < len(line) < 60 and
            re.search(r'[A-Z]', line) and
            not re.match(r'^\d', line) and
            not re.search(r'(remote|india|bangalore|easy apply|\d+k)', line, re.IGNORECASE)):
            return line

    return "Unknown"


def _extract_location(text: str) -> str:
    patterns = [
        re.compile(r'\b(Bangalore|Bengaluru|Mumbai|Delhi|Hyderabad|Chennai|Pune|'
                   r'Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Kochi|'
                   r'Indore|Lucknow|Bhopal|Nagpur|Surat)\b', re.IGNORECASE),
        re.compile(r'\b(remote|work\s+from\s+home|wfh|hybrid|onsite)\b', re.IGNORECASE),
        re.compile(r'\b(India)\b', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    return "Not specified"


def _extract_salary(text: str) -> str:
    patterns = [
        re.compile(r'₹?\s*\d+K?\s*[-–]\s*₹?\s*\d+K?\s*(?:\(Employer\s+Est\.\))?', re.IGNORECASE),
        re.compile(r'\d+\s*[-–]\s*\d+\s*(?:LPA|lpa|lac|lakh)', re.IGNORECASE),
        re.compile(r'(?:salary|ctc)[:\s]+([^\n,]{5,50})', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0).strip()[:100]
    return "Not specified"


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract skills by keyword matching."""
    text_lower = text.lower()
    found = []
    for skill in _VALID_SKILLS_SET:
        if skill in text_lower:
            found.append(skill)
    return _validate_skills(found)


# ── LLM job extraction ─────────────────────────────────────────────────────────

def _llm_extract_jobs(email: dict) -> list[dict]:
    """LLM-based job extraction from cleaned email text."""
    body  = email.get("body","")
    clean = clean_email_body(body, max_chars=4000)

    if not clean or len(clean) < 50:
        return []

    prompt = _JOB_EXTRACT_PROMPT.format(
        subject=email.get("subject",""),
        body=clean[:3500],
    )

    jobs_data = _llm_json(prompt, [])
    if not isinstance(jobs_data, list):
        return []

    # Get links from email
    parsed    = parse_email_html(body)
    job_links = parsed.get("job_links",[])
    all_links = parsed.get("links",[])
    links     = job_links if job_links else all_links

    result = []
    for i, job in enumerate(jobs_data):
        if not isinstance(job, dict):
            continue
        role = str(job.get("role","")).strip()
        if len(role) < 3:
            continue

        link = links[i] if i < len(links) else None
        result.append({
            "role":          role[:100],
            "company":       str(job.get("company","Unknown")).strip()[:100],
            "location":      str(job.get("location","Not specified")).strip()[:100],
            "salary":        str(job.get("salary","Not specified")).strip()[:100],
            "skills":        _validate_skills(job.get("skills",[]) or []),
            "link":          link,
            "all_links":     [link] if link else [],
            "description":   str(job.get("description","")).strip()[:200],
            "email_id":      email.get("id",""),
            "email_subject": email.get("subject",""),
            "source":        "llm",
        })

    return result


def extract_all_jobs_from_email(email: dict) -> list[dict]:
    """
    Extract all jobs from a single email.
    1. Try Glassdoor BS4 structured extraction first
    2. Fall back to LLM extraction
    """
    body = email.get("body","")

    # Try HTML structured extraction for Glassdoor/Naukri emails
    from utils.email_cleaner import is_html as _is_html
    if _is_html(body):
        jobs = _extract_glassdoor_jobs_from_html(body, email)
        if jobs:
            logger.info(f"BS4 found {len(jobs)} jobs from {email.get('id','')[:20]}")
            return jobs

    # LLM fallback
    time.sleep(0.2)
    jobs = _llm_extract_jobs(email)
    logger.info(f"LLM found {len(jobs)} jobs from {email.get('id','')[:20]}")
    return jobs


def extract_jobs_from_all_emails(emails: list[dict]) -> list[dict]:
    """Extract jobs from multiple emails, deduplicate."""
    all_jobs = []
    for email in emails:
        try:
            jobs = extract_all_jobs_from_email(email)
            all_jobs.extend(jobs)
        except Exception as e:
            logger.warning(f"Job extraction failed: {type(e).__name__}")

    # Deduplicate
    seen, unique = set(), []
    for job in all_jobs:
        key = (
            re.sub(r'\s+','',job.get("role","")).lower()[:40],
            re.sub(r'\s+','',job.get("company","")).lower()[:30],
        )
        if key not in seen and key[0] and key[0] != "unknown":
            seen.add(key)
            unique.append(job)

    logger.info(f"Total unique jobs: {len(unique)}")
    return unique


# ── Job-Resume scoring ──────────────────────────────────────────────────────────

def score_jobs(jobs: list[dict], resume: dict) -> list[dict]:
    """Score all jobs against resume, return sorted by score."""
    resume_skills = set(s.lower() for s in _validate_skills(resume.get("skills",[])))
    scored = []

    for job in jobs:
        job_skills = set(s.lower() for s in _validate_skills(job.get("skills",[])))
        matched    = sorted(job_skills & resume_skills)
        missing    = sorted(job_skills - resume_skills)

        # Rule-based baseline score
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
            exp_months = int(resume.get("experience_years",0) * 12)
            prompt     = _MATCH_PROMPT.format(
                role=job.get("role",""),
                company=job.get("company",""),
                location=job.get("location",""),
                job_skills=", ".join(list(job_skills)[:12]),
                name=resume.get("name",""),
                resume_skills=", ".join(list(resume_skills)[:20]),
                exp_months=exp_months,
                exp_years=round(resume.get("experience_years",0),1),
                current_role=resume.get("current_role",""),
                education=resume.get("education","")[:80],
                projects="; ".join(resume.get("projects",[])[:2])[:200],
            )
            time.sleep(0.2)
            data = _llm_json(prompt, {})

            if isinstance(data, dict) and data.get("match_score") is not None:
                score = max(0, min(97, int(data.get("match_score", rule_score))))
                match = {
                    "match_score":           score,
                    "matched_skills":        _validate_skills(data.get("matched_skills", matched)),
                    "missing_skills":        _validate_skills(data.get("missing_skills", missing)),
                    "fit_reason":            str(data.get("fit_reason","")),
                    "strengths":             str(data.get("strengths","")),
                    "gaps":                  str(data.get("gaps","")),
                    "recommendation":        str(data.get("recommendation", _rec(score))),
                    "ready_to_apply":        bool(data.get("ready_to_apply", score >= 50)),
                    "skill_gap_suggestions": data.get("skill_gap_suggestions",[]) or [],
                }
            else:
                raise ValueError("bad response")

        except Exception:
            match = {
                "match_score":           rule_score,
                "matched_skills":        list(matched),
                "missing_skills":        list(missing),
                "fit_reason":            f"Matched {len(matched)}/{len(job_skills)} required skills",
                "strengths":             f"Skills: {', '.join(list(matched)[:4])}",
                "gaps":                  f"Missing: {', '.join(list(missing)[:3])}",
                "recommendation":        _rec(rule_score),
                "ready_to_apply":        rule_score >= 50,
                "skill_gap_suggestions": [
                    {"skill":s,"resource":f"Search '{s}' on Coursera"}
                    for s in list(missing)[:2]
                ],
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
            st.markdown(f"#### 💼 {job.get('role','Unknown')}")
            parts = []
            co = job.get("company","")
            lo = job.get("location","")
            sa = job.get("salary","")
            if co and co not in ("Unknown",""): parts.append(f"🏢 **{co}**")
            if lo and lo != "Not specified":    parts.append(f"📍 {lo}")
            if sa and sa != "Not specified":    parts.append(f"💰 {sa}")
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
            cols = st.columns(min(len(links), 3))
            for i, lnk in enumerate(links[:3]):
                with cols[i]:
                    label = "🔗 Apply Now" if i == 0 else f"🔗 Link {i+1}"
                    st.markdown(
                        f'<a href="{lnk}" target="_blank">'
                        f'<button style="background:#4285F4;color:white;border:none;'
                        f'padding:6px 12px;border-radius:4px;cursor:pointer;width:100%;">'
                        f'{label}</button></a>',
                        unsafe_allow_html=True,
                    )
        st.markdown("---")


def render_job_task_tab():
    st.header("💼 Job Task — Resume & Job Matching")

    # ── Step 1: Resume Upload ─────────────────────────────────────────────────
    st.subheader("📄 Step 1: Upload Your Resume")
    st.caption("Parsed section by section — name from header, experience from work sections, skills from skills/projects sections")

    cu, cs = st.columns([2,2])
    with cu:
        uploaded = st.file_uploader(
            "Upload Resume (PDF or TXT)",
            type=["pdf","txt"],
            key="jt_resume_upload",
        )

    resume = st.session_state.get("jt_resume")

    if uploaded:
        fname = uploaded.name
        if st.session_state.get("jt_resume_fname") != fname or not resume:
            with st.spinner("Parsing resume section by section using AI..."):
                fb     = uploaded.read()
                ft     = "pdf" if fname.lower().endswith(".pdf") else "txt"
                resume = parse_resume(fb, ft)
                st.session_state["jt_resume"]       = resume
                st.session_state["jt_resume_fname"] = fname

    with cs:
        if resume and resume.get("name"):
            st.success(f"✅ **{resume['name']}**")
            exp_months = int(resume.get("experience_years",0) * 12)
            st.caption(
                f"Skills: {len(resume.get('skills',[]))} | "
                f"Experience: {exp_months} months | "
                f"Role: {resume.get('current_role','—')}"
            )
        elif resume:
            st.warning("Resume parsed — some fields may be incomplete")
        else:
            st.info("📤 Upload your resume to get started")

    if resume and (resume.get("name") or resume.get("skills")):
        with st.expander("📋 View Parsed Resume Details", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Name:** {resume.get('name','—')}")
                st.markdown(f"**Current Role:** {resume.get('current_role','—')}")
                exp_months = int(resume.get("experience_years",0) * 12)
                st.markdown(f"**Experience:** {exp_months} months")
                st.markdown(f"**Education:** {resume.get('education','—')}")
                if resume.get("experience"):
                    st.write("**Experience Details:**")
                    st.caption(resume["experience"][:400])
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
                    for p in resume["projects"][:4]:
                        st.caption(f"• {str(p)[:120]}")
                if resume.get("certifications"):
                    st.write("**Certifications:**")
                    for c in resume["certifications"][:3]:
                        st.caption(f"• {c}")
                if resume.get("summary"):
                    st.write("**Summary:**")
                    st.caption(resume["summary"])

    st.divider()

    # ── Step 2: Extract Jobs ──────────────────────────────────────────────────
    st.subheader("🔍 Step 2: Extract Jobs from Your Emails")

    source = "gmail" if st.session_state.get("authenticated") else "demo"
    try:
        from memory.repository import get_all_emails, get_all_processed
        all_emails = get_all_emails(source=source)
        proc_list  = get_all_processed(source=source)
        proc_map   = {p["email_id"]: p for p in proc_list}
    except Exception as e:
        st.error(f"Could not load emails: {type(e).__name__}")
        return

    # Find job emails
    _JOB_SENDERS = [
        "glassdoor","naukri","linkedin","indeed","monster",
        "resume.io","jobs@","career@","hiring@","recruit@",
    ]
    job_emails = []
    for email in all_emails:
        cat    = proc_map.get(email.get("id",""),{}).get("category","")
        sender = email.get("sender","").lower()
        if (cat in {"Job / Recruitment","Job","job_recruitment"} or
            any(js in sender for js in _JOB_SENDERS)):
            job_emails.append(email)

    if not job_emails:
        st.info(
            f"No job emails found among {len(all_emails)} emails. "
            "Make sure emails are processed in the Inbox tab first."
        )
        return

    st.success(f"Found **{len(job_emails)} job emails** — each typically has 5-12 listings")

    c1, c2, c3 = st.columns(3)
    with c1:
        max_e = st.slider("Emails to process", 1, min(len(job_emails),20), min(len(job_emails),5), key="jt_max")
    with c2:
        st.write("")
        extract_btn = st.button("🚀 Extract All Jobs", type="primary", use_container_width=True, key="jt_ext")
    with c3:
        st.write("")
        if st.button("🗑️ Clear", use_container_width=True, key="jt_clr"):
            st.session_state.pop("jt_jobs",None)
            st.session_state.pop("jt_scored",None)
            st.rerun()

    if extract_btn:
        to_proc = job_emails[:max_e]
        prog    = st.progress(0)
        status  = st.empty()
        all_j   = []
        for i, email in enumerate(to_proc):
            status.text(f"Processing {i+1}/{len(to_proc)}: {email.get('subject','')[:50]}")
            jobs = extract_all_jobs_from_email(email)
            all_j.extend(jobs)
            prog.progress((i+1)/len(to_proc))

        # Deduplicate
        seen, unique = set(), []
        for job in all_j:
            key = (
                re.sub(r'\s+','',job.get("role","")).lower()[:40],
                re.sub(r'\s+','',job.get("company","")).lower()[:30],
            )
            if key not in seen and key[0]:
                seen.add(key)
                unique.append(job)

        st.session_state["jt_jobs"]   = unique
        st.session_state.pop("jt_scored",None)
        prog.progress(1.0)
        status.success(f"✅ Found **{len(unique)} unique jobs** from {len(to_proc)} emails!")

    st.divider()

    # ── Step 3: Display and Score ─────────────────────────────────────────────
    extracted = st.session_state.get("jt_jobs",[])
    if not extracted:
        st.info("Click **🚀 Extract All Jobs** above to find job listings from your emails.")
        return

    st.subheader(f"📊 Step 3: {len(extracted)} Jobs Found")

    if resume and resume.get("skills") and not st.session_state.get("jt_scored"):
        if st.button(f"🎯 Score {len(extracted)} Jobs Against My Resume", type="secondary", key="jt_score"):
            with st.spinner("AI scoring in progress..."):
                scored = score_jobs(extracted, resume)
                st.session_state["jt_scored"] = scored
            st.rerun()

    jobs_to_show = st.session_state.get("jt_scored") or extracted
    show_match   = bool(st.session_state.get("jt_scored") and resume and resume.get("skills"))

    m1, m2, m3 = st.columns(3)
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
    elif not resume:
        st.caption("Upload resume above and click Score for match percentages")

    for i, job in enumerate(jobs_to_show):
        _render_job_card(job, i, show_match)