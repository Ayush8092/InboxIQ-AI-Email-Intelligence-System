"""
Production-grade Job Intelligence Service.

Fixes:
1. enrich_job_with_scraping exported (fixes ImportError)
2. Multi-job extraction from single HTML email
3. Each job title is a link - extracted from anchor tags
4. LLM extraction layer for structured JSON
5. Semantic matching via LLM scoring
6. Strong skill validation
7. Fixed 100% match score bug
8. Resume section-based parsing
"""
import re
import json
import time
from functools import lru_cache
from urllib.parse import urlparse
from utils.email_cleaner import parse_email_html, clean_email_body
from utils.llm_client import call_llm
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

_NO_SCRAPE = {
    "glassdoor.com","glassdoor.co.in","linkedin.com","indeed.com",
    "naukri.com","monster.com","shine.com","foundit.in","instahyre.com",
    "ziprecruiter.com","careerbuilder.com",
}

_VALID_SKILLS = {
    "python","java","javascript","typescript","c++","c#","golang","go","rust",
    "swift","kotlin","php","ruby","scala","r programming","matlab","bash",
    "shell scripting","perl","dart","flutter","react","reactjs","angular",
    "vue","vuejs","nextjs","nodejs","node.js","express","django","flask",
    "fastapi","spring boot","html","css","tailwind","bootstrap","graphql",
    "rest api","restful","grpc","websocket","machine learning","deep learning",
    "nlp","natural language processing","computer vision","tensorflow","pytorch",
    "keras","scikit-learn","pandas","numpy","scipy","matplotlib","sql","mysql",
    "postgresql","sqlite","nosql","mongodb","redis","elasticsearch","cassandra",
    "spark","hadoop","kafka","airflow","dbt","tableau","power bi","excel",
    "data analysis","data engineering","etl","mlops","llm","openai","langchain",
    "rag","vector database","embeddings","prompt engineering","aws","azure","gcp",
    "google cloud","docker","kubernetes","ci/cd","jenkins","github actions",
    "terraform","ansible","linux","git","devops","microservices","serverless",
    "system design","oop","agile","scrum","api development","data structures",
    "algorithms","problem solving","object oriented programming","generative ai",
    "transformers","bert","gpt","hugging face","xgboost","lightgbm",
}

_SKILL_ALIASES = {
    "js":"javascript","ts":"typescript","py":"python",
    "ml":"machine learning","dl":"deep learning",
    "k8s":"kubernetes","tf":"tensorflow","cv":"computer vision",
    "rdbms":"sql","ai":"machine learning","gen ai":"generative ai",
}


def _validate_skills(skills: list) -> list[str]:
    if not skills:
        return []
    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s = s.strip().strip("\"'.,;:").strip()
        if len(s) < 2 or len(s) > 60:
            continue
        if re.match(r'^[\d\s\W]+$', s):
            continue
        if re.match(r'^[a-zA-Z]$', s):
            continue
        if re.search(r'[{};:@#<>]', s):
            continue
        s_lower   = s.lower()
        canonical = _SKILL_ALIASES.get(s_lower, s_lower)
        if canonical not in seen:
            seen.add(canonical)
            cleaned.append(s)
    return cleaned[:20]


def _normalize_salary(salary) -> str:
    if not salary:
        return "Not specified"
    s = str(salary).strip()
    if not s or s.lower() in ("none","null","not specified","n/a",""):
        return "Not specified"
    return re.sub(r'\s+', ' ', s)[:100]


def _normalize_location(loc) -> str:
    if not loc:
        return "Not specified"
    s = str(loc).strip()
    if not s or s.lower() in ("none","null","not specified","n/a",""):
        return "Not specified"
    return s[:100]


# ── Primary: BeautifulSoup multi-job extraction ────────────────────────────────

def _bs4_extract_all_jobs(email: dict) -> list[dict]:
    """
    Extract ALL jobs from email HTML using BeautifulSoup.
    Each job title is typically an anchor tag → we extract title + link together.
    This is the core fix: find every <a> tag that looks like a job title.
    """
    body = email.get("body","")
    if not body:
        return []

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(body, "html.parser")

    # Remove noise tags
    for tag in soup(["script","style","head","meta","link","noscript"]):
        tag.decompose()

    jobs = []

    # Strategy 1: Find anchor tags whose text looks like a job title
    # "Software Engineer at Google", "Data Scientist - Infosys", etc.
    job_title_pattern = re.compile(
        r'\b(engineer|developer|analyst|scientist|manager|intern|designer|'
        r'architect|lead|senior|junior|associate|consultant|specialist|'
        r'director|executive|officer|head|vp|president|coordinator|'
        r'fullstack|full.stack|frontend|backend|devops|sde|swe|ml|ai)\b',
        re.IGNORECASE
    )

    seen_titles = set()

    for a in soup.find_all("a", href=True):
        href  = a["href"].strip()
        title = a.get_text(strip=True)

        if not href.startswith(("http://","https://")):
            continue
        if len(title) < 5 or len(title) > 120:
            continue
        if not job_title_pattern.search(title):
            continue

        # Clean title
        title = re.sub(r'\s+', ' ', title).strip()
        title_lower = title.lower()
        if title_lower in seen_titles:
            continue
        seen_titles.add(title_lower)

        # Get surrounding context for company/location/salary
        parent   = a.parent
        context  = ""
        for _ in range(4):  # go up 4 levels
            if parent:
                context = parent.get_text(separator=" ", strip=True)
                if len(context) > 30:
                    break
                parent = parent.parent

        # Parse role and company from title
        role, company = _split_role_company(title)
        if not role:
            role = title

        # Extract location from context
        location = _extract_location_from_text(context)

        # Extract salary from context
        salary = _extract_salary_from_text(context)

        # Extract skills from context
        skills = _extract_skills_from_text(context + " " + title)

        # Build description from context
        desc = context[:200] if context else f"{role} at {company}"

        jobs.append({
            "role":          role[:100],
            "company":       company[:100] if company else "Unknown",
            "location":      location,
            "salary":        salary,
            "skills":        skills,
            "link":          href,
            "all_links":     [href],
            "description":   desc[:200],
            "email_id":      email.get("id",""),
            "email_subject": email.get("subject",""),
            "source":        "bs4_anchor",
        })

    logger.info(f"BS4 anchor extraction: {len(jobs)} jobs from email")

    # Strategy 2: If no anchor-based jobs found, try table rows
    if not jobs:
        rows = soup.find_all("tr")
        for row in rows:
            row_text = row.get_text(separator=" ", strip=True)
            if len(row_text) < 15:
                continue
            if not job_title_pattern.search(row_text):
                continue
            link = None
            for a in row.find_all("a", href=True):
                h = a["href"]
                if h.startswith(("http://","https://")):
                    link = h
                    break
            job = _parse_job_from_text_block(row_text, email, link)
            if job:
                t = job.get("role","").lower()
                if t not in seen_titles:
                    seen_titles.add(t)
                    jobs.append(job)

    return _deduplicate_jobs(jobs)


def _split_role_company(title: str) -> tuple[str, str]:
    """Split 'Role at Company' or 'Role - Company' into (role, company)."""
    # Pattern: X at Y
    m = re.match(r'^(.+?)\s+at\s+(.+?)(?:\s+and\s+\d+.*)?$', title, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern: X - Y or X | Y
    m = re.match(r'^(.+?)\s*[-–|]\s*(.+?)$', title)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return title, ""


def _extract_location_from_text(text: str) -> str:
    patterns = [
        re.compile(r'(?:location|city|place|based\s+in|office)[:\s]+([A-Z][A-Za-z\s,\.]{3,50})', re.IGNORECASE),
        re.compile(r'\b(Bangalore|Bengaluru|Mumbai|Delhi|Hyderabad|Chennai|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Kochi|Indore|Lucknow|Bhopal|Nagpur|Surat|Vadodara)\b', re.IGNORECASE),
        re.compile(r'\b(remote|work\s+from\s+home|wfh|hybrid|onsite|on-site)\b', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            loc = m.group(1) if m.lastindex else m.group(0)
            if len(loc) >= 3:
                return _normalize_location(loc)
    return "Not specified"


def _extract_salary_from_text(text: str) -> str:
    patterns = [
        re.compile(r'(?:salary|ctc|package|compensation)[:\s]*([₹$€]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K|L|per\s+month|pm|pa)?(?:\s*[-–]\s*[₹$€]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K)?)?)', re.IGNORECASE),
        re.compile(r'([₹$]\s*[\d,\.]+(?:\s*[-–]\s*[₹$]\s*[\d,\.]+)?\s*(?:LPA|lpa|K|k))', re.IGNORECASE),
        re.compile(r'(\d+[-–]\d+\s*(?:LPA|lpa|lac|lakh))', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return _normalize_salary(m.group(1))
    return "Not specified"


_SKILL_SECTION_RE = re.compile(
    r'(?:required\s+skills?|skill\s+set|tech(?:nical)?\s+skills?|'
    r'requirements?|qualifications?|must\s+have|key\s+skills?|technologies)'
    r'[:\s]*([^\n]{10,400})',
    re.IGNORECASE
)


def _extract_skills_from_text(text: str) -> list[str]:
    skills = []
    for match in _SKILL_SECTION_RE.findall(text):
        for part in re.split(r'[,;•·|/]', match):
            p = part.strip()
            if 2 <= len(p) <= 40:
                skills.append(p)
    text_lower = text.lower()
    for skill in _VALID_SKILLS:
        if skill in text_lower:
            idx    = text_lower.find(skill)
            actual = text[idx:idx+len(skill)]
            skills.append(actual)
    return _validate_skills(skills)


def _parse_job_from_text_block(text: str, email: dict, link: str | None) -> dict | None:
    if not text or len(text) < 15:
        return None

    role, company = "", ""
    subject = email.get("subject","")

    for src in [subject, text]:
        m = re.match(r'^(.{5,80}?)\s+at\s+(.{3,60}?)(?:\s+and\s+\d+|\s*[-|,].*)?$', src.split("\n")[0], re.IGNORECASE)
        if m:
            role    = m.group(1).strip()[:100]
            company = m.group(2).strip()[:100]
            break
        m = re.match(r'^([A-Z][A-Za-z\s/&+\-\.]{5,60}?)\s*[-–|]\s*([A-Z][A-Za-z\s\.]{3,50})', src)
        if m:
            role    = m.group(1).strip()[:100]
            company = m.group(2).strip()[:100]
            break

    if not role or len(role) < 3:
        return None

    return {
        "role":          role,
        "company":       company or "Unknown",
        "location":      _extract_location_from_text(text),
        "salary":        _extract_salary_from_text(text),
        "skills":        _extract_skills_from_text(text),
        "link":          link,
        "all_links":     [link] if link else [],
        "description":   text[:200],
        "email_id":      email.get("id",""),
        "email_subject": email.get("subject",""),
        "source":        "rule",
    }


def _deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for job in jobs:
        key = (
            re.sub(r'\s+', ' ', job.get("role","")).lower()[:60],
            re.sub(r'\s+', ' ', job.get("company","")).lower()[:40],
        )
        if key not in seen and key[0]:
            seen.add(key)
            unique.append(job)
    return unique


# ── LLM multi-job extraction ───────────────────────────────────────────────────

_LLM_MULTI_JOB_PROMPT = """\
This email contains multiple job listings. Extract ALL of them carefully.

Email subject: {subject}
Email content:
{body}

Extract EVERY distinct job listing. For EACH job return:
{{"role":"exact job title","company":"company name","location":"city or Remote or Not specified","salary":"salary range or Not specified","skills":["python","sql","react"],"description":"1 sentence about role"}}

IMPORTANT RULES:
- Extract ALL jobs, typically 3-10 per email
- skills must be real technologies (python, java, sql, react, etc.) NOT single letters
- If a job title IS a link, include it
- Return [] only if truly no jobs found

Return JSON array only, no preamble:"""


@lru_cache(maxsize=500)
def _cached_llm(prompt: str) -> str:
    return call_llm(prompt, temperature=0.0, max_tokens=2500, use_cache=True)


def _llm_extract_jobs(email: dict) -> list[dict]:
    """LLM-based extraction — fallback when structural extraction misses jobs."""
    body  = email.get("body","")
    clean = clean_email_body(body, max_chars=3000)

    prompt = _LLM_MULTI_JOB_PROMPT.format(
        subject=email.get("subject",""),
        body=clean[:2500],
    )

    try:
        raw = _cached_llm(prompt)
        m   = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            data  = json.loads(m.group(0))
            links = parse_email_html(body).get("job_links",[])
            jobs  = []
            for i, job in enumerate(data):
                if not isinstance(job, dict):
                    continue
                role = str(job.get("role","")).strip()
                if len(role) < 3:
                    continue
                jobs.append({
                    "role":          role[:100],
                    "company":       str(job.get("company","Unknown")).strip()[:100],
                    "location":      _normalize_location(job.get("location","")),
                    "salary":        _normalize_salary(job.get("salary","")),
                    "skills":        _validate_skills(job.get("skills",[])),
                    "link":          links[i] if i < len(links) else None,
                    "all_links":     [links[i]] if i < len(links) else [],
                    "description":   str(job.get("description","")).strip()[:200],
                    "email_id":      email.get("id",""),
                    "email_subject": email.get("subject",""),
                    "source":        "llm",
                })
            logger.info(f"LLM extracted {len(jobs)} jobs")
            return jobs
    except Exception as e:
        logger.warning(f"LLM job extract failed: {type(e).__name__}")
    return []


def extract_jobs_from_email(email: dict) -> list[dict]:
    """
    Extract ALL jobs from a single email.
    1. BS4 anchor-tag based (finds job title links)
    2. LLM fallback
    """
    jobs = _bs4_extract_all_jobs(email)

    if not jobs:
        time.sleep(0.2)
        jobs = _llm_extract_jobs(email)

    return jobs


def enrich_job_with_scraping(job: dict) -> dict:
    """
    Enrich job data via web scraping.
    Skips bot-protected sites (Glassdoor, LinkedIn, etc.)
    This function is EXPORTED — imported by UI layer.
    """
    link = job.get("link")
    if not link:
        return job

    try:
        domain = urlparse(link).netloc.lower().lstrip("www.")
        if any(blocked in domain for blocked in _NO_SCRAPE):
            logger.debug(f"Skipping bot-protected site: {domain}")
            return job
    except Exception:
        return job

    try:
        import requests
        resp = requests.get(
            link,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=6,
            allow_redirects=True,
        )
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type",""):
            return job
        extra_text   = clean_email_body(resp.text, max_chars=2000)
        extra_skills = _extract_skills_from_text(extra_text)
        if extra_skills:
            merged        = list(dict.fromkeys((job.get("skills") or []) + extra_skills))[:25]
            job["skills"] = _validate_skills(merged)
            job["scraped"] = True
        extra_salary = _extract_salary_from_text(extra_text)
        if extra_salary != "Not specified" and job.get("salary") == "Not specified":
            job["salary"] = extra_salary
        extra_loc = _extract_location_from_text(extra_text)
        if extra_loc != "Not specified" and job.get("location") == "Not specified":
            job["location"] = extra_loc
    except Exception as e:
        logger.debug(f"Scrape failed for {link[:50]}: {type(e).__name__}")

    return job


def process_job_emails(
    emails: list[dict],
    enrich: bool = False,
    max_emails: int = 30,
) -> list[dict]:
    all_jobs = []
    for email in emails[:max_emails]:
        try:
            jobs = extract_jobs_from_email(email)
            if enrich:
                jobs = [enrich_job_with_scraping(j) for j in jobs]
                time.sleep(0.3)
            all_jobs.extend(jobs)
        except Exception as e:
            logger.warning(f"Job extraction error: {type(e).__name__}")

    # Final dedup: (role, company, location)
    seen, unique = set(), []
    for job in all_jobs:
        key = (
            re.sub(r'\s+', ' ', job.get("role","")).lower()[:60],
            re.sub(r'\s+', ' ', job.get("company","")).lower()[:40],
            job.get("location","").lower()[:30],
        )
        if key not in seen and key[0]:
            seen.add(key)
            unique.append(job)

    logger.info(f"Total unique jobs: {len(unique)} from {len(emails)} emails")
    return unique


# ── Resume parsing ──────────────────────────────────────────────────────────────

_RESUME_PARSE_PROMPT = """\
Parse this resume carefully and extract all information.

{resume_text}

Return this JSON structure only, no explanation:
{{
  "name": "full name (NOT truncated)",
  "skills": ["python", "machine learning", "sql", "react"],
  "experience_years": 1,
  "experience": "2-3 sentence summary of ALL work experience with dates",
  "current_role": "most recent job title",
  "education": "degree, field, institution, year",
  "projects": ["project name: brief description and tech used"],
  "certifications": ["certification name and issuer"],
  "summary": "2-3 sentence professional summary based on the resume"
}}

CRITICAL:
- name must be the COMPLETE full name (e.g. "Ayush Kumar" not "Ayu")
- skills must be real technologies, minimum 3 characters each
- experience_years: count actual years/months from work history
- extract ALL projects listed"""


def _clean_resume_text(raw: str) -> str:
    text = raw.replace("\r\n","\n").replace("\r","\n")
    lines = text.split("\n")
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            fixed.append("")
            i += 1
            continue
        # Join broken words
        if (i + 1 < len(lines) and
            line and line[-1].isalpha() and
            lines[i+1].strip() and
            lines[i+1].strip()[0].islower() and
            len(line) < 30):
            fixed.append(line + " " + lines[i+1].strip())
            i += 2
        else:
            fixed.append(line)
            i += 1
    text = "\n".join(fixed)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\w{3,})\s+(ously|tion|ing|ment|ness|ity|ance|ence|ible|able)\b', r'\1\2', text)
    return text.strip()


def _extract_sections(text: str) -> dict:
    headers = [
        "professional summary","summary","objective","profile",
        "education","academic","qualification",
        "experience","work experience","internship","internships","employment",
        "technical skills","skills","core competencies","technologies",
        "projects","project work","personal projects",
        "certifications","achievements","awards","honors",
        "languages","interests","hobbies",
    ]
    pattern = re.compile(
        r'(?:^|\n)(' + "|".join(re.escape(h) for h in headers) + r')[\s:]*\n',
        re.IGNORECASE
    )
    sections    = {}
    last_header = "intro"
    last_pos    = 0
    for m in pattern.finditer(text):
        if last_pos < m.start():
            sections[last_header] = text[last_pos:m.start()].strip()
        last_header = m.group(1).lower().strip()
        last_pos    = m.end()
    sections[last_header] = text[last_pos:].strip()
    return sections


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        import pdfminer.high_level as pmh, io
        text = pmh.extract_text(io.BytesIO(file_bytes)) or ""
    except Exception:
        pass
    if len(text.strip()) < 100:
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            pass
    if len(text.strip()) < 100:
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    return _clean_resume_text(text)


def parse_resume(file_bytes: bytes, file_type: str = "pdf") -> dict:
    raw_text = (
        extract_text_from_pdf(file_bytes)
        if file_type == "pdf"
        else _clean_resume_text(file_bytes.decode("utf-8", errors="ignore"))
    )
    if len(raw_text.strip()) < 100:
        logger.warning("Resume text too short")
        return _empty_resume()

    rule_skills = _extract_skills_from_text(raw_text)
    sections    = _extract_sections(raw_text)

    # Build structured context
    context_parts = []
    for sec_name, sec_content in sections.items():
        if sec_content and len(sec_content) > 10:
            context_parts.append(f"=== {sec_name.upper()} ===\n{sec_content}")
    context = "\n\n".join(context_parts) if len(context_parts) > 2 else raw_text

    try:
        prompt = _RESUME_PARSE_PROMPT.format(resume_text=context[:3500])
        raw    = _cached_llm(prompt)
        m      = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data      = json.loads(m.group(0))
            llm_sk    = _validate_skills(data.get("skills",[]))
            merged    = list(dict.fromkeys(llm_sk + rule_skills))[:30]
            name      = str(data.get("name","")).strip()
            if len(name) < 2:
                name = raw_text.split("\n")[0].strip()[:60]
            return {
                "name":             name,
                "skills":           merged,
                "experience_years": int(data.get("experience_years") or 0),
                "experience":       str(data.get("experience","")),
                "current_role":     str(data.get("current_role","")),
                "education":        str(data.get("education","")),
                "projects":         data.get("projects",[]) or [],
                "certifications":   data.get("certifications",[]) or [],
                "summary":          str(data.get("summary","")),
                "raw_text_length":  len(raw_text),
            }
    except Exception as e:
        logger.warning(f"Resume LLM parse failed: {type(e).__name__}")

    name = raw_text.split("\n")[0].strip()[:60]
    return {
        "name":name,"skills":rule_skills,"experience_years":0,
        "experience":"","current_role":"","education":"",
        "projects":[],"certifications":[],
        "summary":f"Resume with {len(rule_skills)} skills detected",
        "raw_text_length":len(raw_text),
    }


def _empty_resume() -> dict:
    return {
        "name":"","skills":[],"experience_years":0,"experience":"",
        "current_role":"","education":"","projects":[],"certifications":[],
        "summary":"","raw_text_length":0,
    }


# ── Semantic job-resume scoring ────────────────────────────────────────────────

_MATCH_PROMPT = """\
Score this job-resume match realistically. Return JSON only.

JOB:
Title: {role}
Company: {company}
Required Skills: {job_skills}
Description: {description}

CANDIDATE:
Skills: {resume_skills}
Experience: {experience_years} years
Current Role: {current_role}
Education: {education}
Summary: {summary}

Scoring criteria (be REALISTIC, NOT inflated):
- Skill overlap: up to 50 points
- Role/domain relevance: up to 30 points
- Experience fit: up to 20 points

{{
  "match_score": 65,
  "matched_skills": ["python", "sql"],
  "missing_skills": ["kubernetes", "golang"],
  "strengths": "specific reason this candidate is a good fit",
  "gaps": "specific skills or experience missing",
  "recommendation": "Good Match",
  "fit_reason": "You have strong Python and data skills but lack cloud experience",
  "ready_to_apply": true,
  "skill_gap_suggestions": [
    {{"skill": "kubernetes", "resource": "Kubernetes Basics on Coursera"}}
  ]
}}

RULES:
- match_score: integer 0-97 (NEVER 100 unless absolutely perfect)
- recommendation: one of "Strong Match", "Good Match", "Partial Match", "Weak Match"
- ready_to_apply: true if score >= 55"""


def score_job_match(job: dict, resume: dict) -> dict:
    job_skills    = {s.lower() for s in _validate_skills(job.get("skills",[]))}
    resume_skills = {s.lower() for s in _validate_skills(resume.get("skills",[]))}

    if not job_skills:
        rule_score  = 40
        matched, missing = [], []
    else:
        matched_set = job_skills & resume_skills
        missing_set = job_skills - resume_skills
        matched     = sorted(matched_set)
        missing     = sorted(missing_set)
        overlap     = (len(matched_set) / max(len(job_skills), 1)) * 50
        role_score  = _role_relevance(job, resume) * 30
        rule_score  = min(97, int(overlap + role_score + 20))

    def _rec(s):
        if s >= 80: return "Strong Match"
        if s >= 60: return "Good Match"
        if s >= 40: return "Partial Match"
        return "Weak Match"

    if resume_skills:
        try:
            prompt = _MATCH_PROMPT.format(
                role=job.get("role",""),
                company=job.get("company",""),
                job_skills=", ".join(list(job_skills)[:15]),
                description=job.get("description",""),
                resume_skills=", ".join(list(resume_skills)[:20]),
                experience_years=resume.get("experience_years",0),
                current_role=resume.get("current_role",""),
                education=resume.get("education",""),
                summary=resume.get("summary","")[:200],
            )
            time.sleep(0.2)
            raw = _cached_llm(prompt)
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data  = json.loads(m.group(0))
                score = max(0, min(97, int(data.get("match_score", rule_score))))
                return {
                    "match_score":           score,
                    "matched_skills":        _validate_skills(data.get("matched_skills", matched)),
                    "missing_skills":        _validate_skills(data.get("missing_skills", missing)),
                    "strengths":             str(data.get("strengths","")),
                    "gaps":                  str(data.get("gaps","")),
                    "recommendation":        str(data.get("recommendation", _rec(score))),
                    "fit_reason":            str(data.get("fit_reason","")),
                    "ready_to_apply":        bool(data.get("ready_to_apply", score >= 55)),
                    "skill_gap_suggestions": data.get("skill_gap_suggestions",[]) or [],
                }
        except Exception as e:
            logger.warning(f"Match LLM failed: {type(e).__name__}")

    return {
        "match_score":           rule_score,
        "matched_skills":        list(matched),
        "missing_skills":        list(missing),
        "strengths":             f"Matched {len(matched)}/{len(job_skills)} skills",
        "gaps":                  f"Missing: {', '.join(list(missing)[:5])}",
        "recommendation":        _rec(rule_score),
        "fit_reason":            f"Skill overlap: {len(matched)}/{len(job_skills)} required skills",
        "ready_to_apply":        rule_score >= 55,
        "skill_gap_suggestions": [
            {"skill":s,"resource":f"Search '{s}' on Coursera or Udemy"}
            for s in list(missing)[:3]
        ],
    }


def _role_relevance(job: dict, resume: dict) -> float:
    jt = set(re.findall(r'\b\w{3,}\b', job.get("role","").lower()))
    rt = set(re.findall(r'\b\w{3,}\b', resume.get("current_role","").lower()))
    if not jt:
        return 0.5
    return min(1.0, len(jt & rt) / len(jt))


def score_all_jobs(jobs: list[dict], resume: dict) -> list[dict]:
    scored = []
    for job in jobs:
        try:
            score = score_job_match(job, resume)
            scored.append({**job, "match": score})
        except Exception as e:
            logger.warning(f"Score failed: {type(e).__name__}")
            scored.append({**job, "match": {"match_score": 0}})
    scored.sort(key=lambda x: x.get("match",{}).get("match_score",0), reverse=True)
    return scored