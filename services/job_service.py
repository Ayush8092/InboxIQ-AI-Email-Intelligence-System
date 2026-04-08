"""
Production-grade Job Intelligence Service.

Architecture:
  1. HTML link extraction FIRST (before any text processing)
  2. Rule-based job parsing (zero LLM cost, fast)
  3. LLM only as fallback for unstructured emails
  4. Strong skill validation (no single-char skills)
  5. Strict deduplication on (role, company, location)
  6. Bot-protected site skipping
  7. @lru_cache on LLM calls
  8. 100% match bug fixed — proper scoring algorithm
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

# ── Sites that block scrapers — skip entirely ──────────────────────────────────
_NO_SCRAPE = {
    "glassdoor.com", "glassdoor.co.in", "linkedin.com",
    "indeed.com", "naukri.com", "monster.com",
    "shine.com", "foundit.in", "instahyre.com",
    "ziprecruiter.com", "careerbuilder.com",
}

# ── Valid skill whitelist — prevents single chars like "r", "c" ───────────────
_VALID_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang",
    "go", "rust", "swift", "kotlin", "php", "ruby", "scala", "r programming",
    "matlab", "bash", "shell scripting", "perl", "dart", "flutter",
    # Web
    "react", "reactjs", "angular", "angularjs", "vue", "vuejs", "nextjs",
    "nodejs", "node.js", "express", "expressjs", "django", "flask", "fastapi",
    "spring boot", "laravel", "rails", "html", "css", "tailwind", "bootstrap",
    "graphql", "rest api", "restful", "soap", "grpc", "websocket",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "sql", "mysql", "postgresql", "sqlite", "nosql", "mongodb",
    "redis", "elasticsearch", "cassandra", "dynamodb",
    "spark", "hadoop", "kafka", "airflow", "dbt", "flink",
    "tableau", "power bi", "looker", "excel", "data analysis",
    "data engineering", "etl", "data pipeline", "feature engineering",
    "model deployment", "mlops", "llm", "openai", "langchain", "rag",
    "vector database", "embeddings", "fine-tuning", "prompt engineering",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "terraform",
    "ansible", "linux", "unix", "git", "devops", "sre", "microservices",
    "serverless", "lambda", "cloud formation",
    # General Engineering
    "data structures", "algorithms", "system design", "oop",
    "design patterns", "agile", "scrum", "jira", "api development",
    "object oriented programming", "problem solving",
}

# Map common abbreviations to canonical names
_SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "ml": "machine learning", "dl": "deep learning",
    "k8s": "kubernetes", "tf": "tensorflow",
    "cv": "computer vision", "nn": "neural networks",
    "rdbms": "sql", "oop": "object oriented programming",
}


def _validate_skills(skills: list) -> list[str]:
    """
    Validate and clean skill list.
    Removes: single chars, random strings, numbers, CSS properties.
    """
    if not skills:
        return []

    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s = s.strip().strip("\"'.,;").strip()

        # Length guard — min 2 chars, max 50
        if len(s) < 2 or len(s) > 50:
            continue

        # Skip obvious noise
        if re.match(r'^[\d\s\W]+$', s):           # only digits/symbols
            continue
        if re.match(r'^[a-zA-Z]$', s):             # single letter like "r", "c"
            continue
        if re.search(r'[{};:@#]', s):              # CSS/code fragments
            continue
        if re.match(r'^\d+px', s, re.IGNORECASE):  # CSS sizes
            continue

        # Normalize
        s_lower = s.lower()
        canonical = _SKILL_ALIASES.get(s_lower, s_lower)

        # Accept if in whitelist OR if it looks like a real tech name
        in_whitelist = canonical in _VALID_SKILLS or s_lower in _VALID_SKILLS
        looks_real   = (
            len(s) >= 3 and
            re.match(r'^[A-Za-z][A-Za-z0-9\s\+\#\-\.\/]+$', s) and
            not s_lower.startswith(("the ", "and ", "for ", "with "))
        )

        if (in_whitelist or looks_real) and canonical not in seen:
            seen.add(canonical)
            # Use canonical name if available, else original
            display = next(
                (k for k, v in _SKILL_ALIASES.items() if v == canonical),
                s
            )
            cleaned.append(s)  # keep original casing

    return cleaned[:20]


def _normalize_salary(salary) -> str:
    if not salary:
        return "Not specified"
    s = str(salary).strip()
    if not s or s.lower() in ("none", "null", "not specified", "n/a", ""):
        return "Not specified"
    # Remove excessive whitespace
    s = re.sub(r'\s+', ' ', s)
    # Standardize currency
    s = s.replace("Rs.", "₹").replace("INR", "₹")
    return s[:100]


def _normalize_location(loc) -> str:
    if not loc:
        return "Not specified"
    s = str(loc).strip()
    if not s or s.lower() in ("none", "null", "not specified", "n/a", ""):
        return "Not specified"
    return s[:100]


# ── Link extraction (from parsed HTML) ────────────────────────────────────────

def _extract_apply_links(email: dict) -> list[str]:
    """
    Extract apply links from email HTML body.
    Uses parse_email_html which properly reads <a href> tags.
    """
    body   = email.get("body", "")
    parsed = parse_email_html(body)

    # Prefer job-specific links
    job_links = parsed.get("job_links", [])
    all_links = parsed.get("links", [])

    # Also look for apply-specific patterns in all links
    apply_links = []
    for link in all_links:
        link_lower = link.lower()
        if any(p in link_lower for p in [
            "/apply", "/job", "/career", "/opening",
            "/position", "/role", "apply?", "jobid",
            "job_id", "jid=",
        ]):
            if link not in apply_links:
                apply_links.append(link)

    # Combine: job domain links first, then apply-pattern links
    combined = list(dict.fromkeys(job_links + apply_links + all_links[:5]))
    return combined[:5]


# ── Rule-based job extraction ─────────────────────────────────────────────────

_ROLE_RE = [
    # "Role at Company" pattern
    re.compile(
        r'^(.{5,80}?)\s+at\s+(.{3,50}?)(?:\s+and\s+\d+\s+more)?(?:\s*[-|,].*)?$',
        re.IGNORECASE
    ),
    # "Role - Company" pattern
    re.compile(r'^(.{5,80}?)\s*[-–|]\s*(.{3,50}?)(?:\s*[-|,].*)?$'),
    # "Apply for Role" pattern
    re.compile(r'apply\s+(?:for\s+)?(.{5,80}?)\s+(?:at|@)\s+(.{3,50})', re.IGNORECASE),
    # "Role position is open" pattern
    re.compile(r'(.{5,60}?)\s+(?:position|role|opening)\s+(?:is\s+)?(?:open|available)', re.IGNORECASE),
]

_SALARY_RE = [
    re.compile(
        r'(?:salary|ctc|package|compensation|pay)[:\s]*'
        r'([₹$€£]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K|L|per\s+month|pm|pa|per\s+annum)?'
        r'(?:\s*[-–to]+\s*[₹$€£]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K|L)?)?)',
        re.IGNORECASE
    ),
    re.compile(r'([₹$]\s*[\d,\.]+(?:\s*[-–]\s*[₹$]\s*[\d,\.]+)?\s*(?:LPA|lpa|K|k))', re.IGNORECASE),
]

_LOCATION_RE = [
    re.compile(r'(?:location|city|place|based in|office)[:\s]+([A-Z][A-Za-z\s,\.]{3,50})', re.IGNORECASE),
    re.compile(r'\b(Bangalore|Mumbai|Delhi|Hyderabad|Chennai|Pune|Kolkata|Noida|Gurgaon|Ahmedabad|Jaipur|remote|work from home|wfh|hybrid)\b', re.IGNORECASE),
]

_SKILL_SECTION_RE = re.compile(
    r'(?:required\s+skills?|skill\s+set|tech(?:nical)?\s+skills?|requirements?|qualifications?)'
    r'[:\s]*([^\n]{10,300})',
    re.IGNORECASE
)


def _rule_extract_from_subject(subject: str) -> tuple[str, str]:
    """Extract role and company from email subject line."""
    role, company = "", ""
    for pattern in _ROLE_RE:
        m = pattern.match(subject.strip())
        if m:
            r = m.group(1).strip()
            c = m.group(2).strip() if m.lastindex >= 2 else ""
            # Clean up
            r = re.sub(r'\s+', ' ', r).strip()
            c = re.sub(r'\s+', ' ', c).strip()
            # Validate role looks like a job title
            if len(r) >= 3 and not r.lower().startswith(("http", "www", "dear")):
                role    = r[:100]
                company = c[:100]
                break
    return role, company


def _rule_extract_skills(text: str) -> list[str]:
    """Extract skills using section detection + keyword matching."""
    skills = []

    # Look for skills section
    skill_matches = _SKILL_SECTION_RE.findall(text)
    for match in skill_matches:
        parts = re.split(r'[,;•·|/]', match)
        for part in parts:
            part = part.strip()
            if 2 <= len(part) <= 40:
                skills.append(part)

    # Keyword matching against known skills
    text_lower = text.lower()
    for skill in _VALID_SKILLS:
        if skill in text_lower:
            # Find actual casing in text
            idx = text_lower.find(skill)
            actual = text[idx: idx + len(skill)]
            skills.append(actual)

    return _validate_skills(skills)


def _rule_extract_job(email: dict) -> dict | None:
    """
    Rule-based extraction — returns one job dict or None.
    Fast, zero LLM cost.
    """
    subject  = email.get("subject", "")
    body_raw = email.get("body", "")
    clean    = clean_email_body(body_raw, max_chars=3000)
    text     = subject + "\n" + clean

    role, company = _rule_extract_from_subject(subject)
    if not role or len(role) < 3:
        return None

    # Salary
    salary = "Not specified"
    for pat in _SALARY_RE:
        m = pat.search(text)
        if m:
            salary = _normalize_salary(m.group(1))
            break

    # Location
    location = "Not specified"
    for pat in _LOCATION_RE:
        m = pat.search(text)
        if m:
            loc = m.group(1) if m.lastindex else m.group(0)
            location = _normalize_location(loc)
            break
    if location == "Not specified":
        if re.search(r'\b(remote|work from home|wfh)\b', text, re.IGNORECASE):
            location = "Remote"
        elif re.search(r'\bhybrid\b', text, re.IGNORECASE):
            location = "Hybrid"

    # Skills
    skills = _rule_extract_skills(text)

    # Links
    links = _extract_apply_links(email)

    # Description from body
    desc_sentences = [
        s.strip() for s in clean.replace("\n", " ").split(".")
        if len(s.strip()) > 30 and len(s.strip()) < 200
    ]
    description = desc_sentences[0] + "." if desc_sentences else f"{role} at {company}"

    return {
        "role":          role,
        "company":       company or "Unknown",
        "location":      location,
        "salary":        salary,
        "skills":        skills,
        "link":          links[0] if links else None,
        "all_links":     links,
        "description":   description[:200],
        "email_id":      email.get("id", ""),
        "email_subject": subject,
        "source":        "rule",
    }


# ── LLM prompts ────────────────────────────────────────────────────────────────

_JOB_EXTRACT_PROMPT = """\
Extract ALL distinct job listings from this email. Return JSON array only, no explanation.

Subject: {subject}
Email content: {body}

For EACH job:
{{"role":"exact job title","company":"company name","location":"city or Remote","salary":"range or Not specified","skills":["python","sql"],"description":"1 sentence about the role"}}

IMPORTANT:
- skills must be real technologies (python, java, sql, etc.) NOT single letters
- If salary not mentioned: "Not specified"
- Return [] if no jobs found

JSON array only:"""

_RESUME_PARSE_PROMPT = """\
Parse this resume text. Return JSON only, no explanation.

{resume_text}

{{"name":"full name or empty","skills":["python","machine learning","sql"],"experience_years":2,"experience":"2-3 sentence work summary","current_role":"most recent title","education":"degree and field","projects":["project name and tech"],"certifications":["cert name"],"summary":"2 sentence professional summary"}}

IMPORTANT: skills must be real technologies, minimum 3 characters each."""

_MATCH_PROMPT = """\
Calculate job-resume match score (0-100). Return JSON only.

Job: {role} at {company}
Required skills: {job_skills}
Job description: {description}

Candidate:
Skills: {resume_skills}
Experience: {resume_experience} years
Current role: {current_role}

Scoring weights:
- Skill overlap: 50%
- Role relevance: 30%
- Experience: 20%

{{"match_score":75,"matched_skills":["python","sql"],"missing_skills":["java"],"strengths":"specific strengths","gaps":"specific gaps","recommendation":"Good Match","fit_reason":"You match X and Y strongly","ready_to_apply":true,"skill_gap_suggestions":[{{"skill":"java","resource":"Java on Coursera"}}]}}

Score MUST be 0-100. Do not give 100 unless perfect match."""


@lru_cache(maxsize=300)
def _cached_llm(prompt: str) -> str:
    """LRU-cached LLM call — identical prompts return cached result."""
    return call_llm(prompt, temperature=0.0, max_tokens=1500, use_cache=True)


def _llm_extract_jobs(email: dict) -> list[dict]:
    """LLM-based extraction — fallback when rule extraction fails."""
    subject  = email.get("subject", "")
    body_raw = email.get("body", "")
    clean    = clean_email_body(body_raw, max_chars=2000)

    prompt = _JOB_EXTRACT_PROMPT.format(
        subject=subject,
        body=clean[:1500],
    )

    try:
        raw = _cached_llm(prompt)
        m   = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                links = _extract_apply_links(email)
                jobs  = []
                for job in data:
                    if not isinstance(job, dict):
                        continue
                    role = str(job.get("role", "")).strip()
                    if len(role) < 3:
                        continue

                    raw_skills     = job.get("skills", [])
                    validated_skills = _validate_skills(raw_skills)

                    jobs.append({
                        "role":          role[:100],
                        "company":       str(job.get("company", "Unknown")).strip()[:100],
                        "location":      _normalize_location(job.get("location", "")),
                        "salary":        _normalize_salary(job.get("salary", "")),
                        "skills":        validated_skills,
                        "link":          links[0] if links else None,
                        "all_links":     links,
                        "description":   str(job.get("description", "")).strip()[:200],
                        "email_id":      email.get("id", ""),
                        "email_subject": email.get("subject", ""),
                        "source":        "llm",
                    })
                return jobs
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM job extract failed: {type(e).__name__}")

    return []


def extract_jobs_from_email(email: dict) -> list[dict]:
    """
    Extract jobs from a single email.
    Strategy: rule-based first → LLM fallback.
    Rate limited: 0.2s between calls.
    """
    # Rule-based first
    rule_job = _rule_extract_job(email)
    if rule_job:
        return [rule_job]

    # LLM fallback
    time.sleep(0.2)
    return _llm_extract_jobs(email)


def _should_skip_scrape(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(blocked in domain for blocked in _NO_SCRAPE)
    except Exception:
        return True


def scrape_job_page(url: str, max_chars: int = 2000) -> str:
    """Lightweight scraping — skips bot-protected sites."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    if _should_skip_scrape(url):
        logger.debug(f"Skipping bot-protected: {url[:60]}")
        return ""
    try:
        import requests
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
            allow_redirects=True,
        )
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return ""
        clean = clean_email_body(resp.text, max_chars=max_chars)
        return clean
    except Exception as e:
        logger.debug(f"Scrape failed {url[:50]}: {type(e).__name__}")
        return ""


def enrich_job_with_scraping(job: dict) -> dict:
    """Enrich via scraping — skip if bot-protected."""
    link = job.get("link")
    if not link or _should_skip_scrape(link):
        return job

    scraped = scrape_job_page(link)
    if not scraped:
        return job

    extra = _rule_extract_skills(scraped)
    if extra:
        merged        = list(dict.fromkeys((job.get("skills") or []) + extra))[:25]
        job["skills"] = _validate_skills(merged)
        job["scraped"] = True

    if job.get("salary") == "Not specified":
        for pat in _SALARY_RE:
            m = pat.search(scraped)
            if m:
                job["salary"] = _normalize_salary(m.group(1))
                break

    return job


def process_job_emails(
    emails: list[dict],
    enrich: bool = False,
    max_emails: int = 20,
) -> list[dict]:
    """
    Process list of job emails.
    Returns deduplicated list sorted by extraction quality.
    """
    all_jobs = []
    for email in emails[:max_emails]:
        try:
            jobs = extract_jobs_from_email(email)
            if enrich:
                jobs = [enrich_job_with_scraping(j) for j in jobs]
                time.sleep(0.3)
            all_jobs.extend(jobs)
        except Exception as e:
            logger.warning(f"Job extraction failed for {email.get('id')}: {type(e).__name__}")

    # Deduplicate: (role_lower, company_lower, location_lower)
    seen, unique = set(), []
    for job in all_jobs:
        role     = re.sub(r'\s+', ' ', job.get("role", "")).lower().strip()[:60]
        company  = re.sub(r'\s+', ' ', job.get("company", "")).lower().strip()[:40]
        location = job.get("location", "").lower().strip()[:30]
        key = (role, company, location)
        if key not in seen and role not in ("", "unknown"):
            seen.add(key)
            unique.append(job)

    logger.info(f"Extracted {len(unique)} unique jobs from {len(emails)} emails")
    return unique


# ── Resume parsing ─────────────────────────────────────────────────────────────

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

    return text.strip()


def parse_resume(file_bytes: bytes, file_type: str = "pdf") -> dict:
    """Parse resume with rule-based skill extraction + LLM structure."""
    raw_text = (
        extract_text_from_pdf(file_bytes)
        if file_type == "pdf"
        else file_bytes.decode("utf-8", errors="ignore")
    )
    raw_text = raw_text.strip()

    # Guard against garbage extraction
    if len(raw_text) < 200:
        logger.warning(f"Resume text too short ({len(raw_text)} chars)")
        return _empty_resume()

    # Rule-based skill extraction first
    rule_skills = _rule_extract_skills(raw_text)

    try:
        prompt = _RESUME_PARSE_PROMPT.format(resume_text=raw_text[:3000])
        raw    = _cached_llm(prompt)
        m      = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data   = json.loads(m.group(0))
            llm_sk = _validate_skills(data.get("skills", []))
            merged = list(dict.fromkeys(llm_sk + rule_skills))[:30]
            return {
                "name":             str(data.get("name", "")),
                "skills":           merged,
                "experience_years": int(data.get("experience_years") or 0),
                "experience":       str(data.get("experience", "")),
                "current_role":     str(data.get("current_role", "")),
                "education":        str(data.get("education", "")),
                "projects":         data.get("projects", []) or [],
                "certifications":   data.get("certifications", []) or [],
                "summary":          str(data.get("summary", "")),
                "raw_text_length":  len(raw_text),
            }
    except Exception as e:
        logger.warning(f"Resume LLM parse failed: {type(e).__name__}")

    return {
        "name": "", "skills": rule_skills, "experience_years": 0,
        "experience": "", "current_role": "", "education": "",
        "projects": [], "certifications": [],
        "summary": f"Resume with {len(rule_skills)} skills",
        "raw_text_length": len(raw_text),
    }


def _empty_resume() -> dict:
    return {
        "name": "", "skills": [], "experience_years": 0, "experience": "",
        "current_role": "", "education": "", "projects": [],
        "certifications": [], "summary": "", "raw_text_length": 0,
    }


# ── Match scoring (fixed 100% bug) ────────────────────────────────────────────

def score_job_match(job: dict, resume: dict) -> dict:
    """
    Score job-resume match.
    
    Fix for 100% bug:
    - Uses weighted scoring NOT just skill overlap
    - Caps at realistic values
    - LLM prompt explicitly instructs not to give 100 for imperfect matches
    """
    job_skills    = {s.lower() for s in _validate_skills(job.get("skills", []))}
    resume_skills = {s.lower() for s in _validate_skills(resume.get("skills", []))}

    # Handle empty skills gracefully
    if not job_skills:
        # Can't score without job skills — give moderate score
        rule_score = 45
        matched, missing = [], []
    else:
        matched_set = job_skills & resume_skills
        missing_set = job_skills - resume_skills
        matched     = sorted(matched_set)
        missing     = sorted(missing_set)

        # Weighted rule score:
        # 50% skill overlap + 30% role relevance (heuristic) + 20% base
        overlap_score = (len(matched_set) / len(job_skills)) * 50
        role_score    = _role_relevance(job, resume) * 30
        base_score    = 20
        rule_score    = min(97, int(overlap_score + role_score + base_score))

    def _rec(score: int) -> str:
        if score >= 80: return "Strong Match"
        if score >= 60: return "Good Match"
        if score >= 40: return "Partial Match"
        return "Weak Match"

    # LLM for richer output
    if job_skills and resume_skills:
        try:
            prompt = _MATCH_PROMPT.format(
                role=job.get("role", ""),
                company=job.get("company", ""),
                job_skills=", ".join(list(job_skills)[:15]),
                description=job.get("description", ""),
                resume_skills=", ".join(list(resume_skills)[:20]),
                resume_experience=resume.get("experience_years", 0),
                current_role=resume.get("current_role", ""),
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
                    "strengths":             str(data.get("strengths", "")),
                    "gaps":                  str(data.get("gaps", "")),
                    "recommendation":        str(data.get("recommendation", _rec(score))),
                    "fit_reason":            str(data.get("fit_reason", "")),
                    "ready_to_apply":        bool(data.get("ready_to_apply", score >= 60)),
                    "skill_gap_suggestions": data.get("skill_gap_suggestions", []) or [],
                }
        except Exception as e:
            logger.warning(f"Match LLM failed: {type(e).__name__}")

    return {
        "match_score":           rule_score,
        "matched_skills":        list(matched),
        "missing_skills":        list(missing),
        "strengths":             f"Matched {len(matched)} of {len(job_skills)} required skills",
        "gaps":                  f"Missing: {', '.join(list(missing)[:5])}",
        "recommendation":        _rec(rule_score),
        "fit_reason":            f"Skill overlap: {len(matched)}/{len(job_skills)} skills",
        "ready_to_apply":        rule_score >= 60,
        "skill_gap_suggestions": [
            {"skill": s, "resource": f"Search '{s}' on Coursera or Udemy"}
            for s in list(missing)[:3]
        ],
    }


def _role_relevance(job: dict, resume: dict) -> float:
    """
    Heuristic role similarity score 0-1.
    Compares job role with candidate's current role.
    """
    job_role     = job.get("role", "").lower()
    resume_role  = resume.get("current_role", "").lower()
    if not job_role or not resume_role:
        return 0.5   # neutral

    # Tokenize
    job_tokens    = set(re.findall(r'\b\w{3,}\b', job_role))
    resume_tokens = set(re.findall(r'\b\w{3,}\b', resume_role))

    if not job_tokens:
        return 0.5

    overlap = len(job_tokens & resume_tokens)
    return min(1.0, overlap / len(job_tokens))


def score_all_jobs(jobs: list[dict], resume: dict) -> list[dict]:
    """Score and sort all jobs by match score."""
    scored = []
    for job in jobs:
        try:
            score = score_job_match(job, resume)
            scored.append({**job, "match": score})
        except Exception as e:
            logger.warning(f"Score failed for {job.get('role')}: {type(e).__name__}")
            scored.append({**job, "match": {"match_score": 0}})

    scored.sort(key=lambda x: x.get("match", {}).get("match_score", 0), reverse=True)
    return scored