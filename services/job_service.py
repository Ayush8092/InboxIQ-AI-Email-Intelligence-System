"""
Production-grade Job Intelligence Service.

Key fixes:
1. Parses FULL HTML email body (finds all jobs in email, not just 1)
2. Detects repeating job blocks (glassdoor/naukri format)
3. Extracts apply links from anchor tags
4. Strong skill validation (no single-char garbage)
5. Fixed 100% match score bug
6. BeautifulSoup for proper HTML parsing
7. LRU cache on all LLM calls
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

# Bot-protected — never scrape these
_NO_SCRAPE = {
    "glassdoor.com", "glassdoor.co.in", "linkedin.com",
    "indeed.com", "naukri.com", "monster.com",
    "shine.com", "foundit.in", "instahyre.com",
    "ziprecruiter.com", "careerbuilder.com",
}

# Canonical skills whitelist
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
    "algorithms","problem solving","object oriented programming",
}

_SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "ml": "machine learning", "dl": "deep learning",
    "k8s": "kubernetes", "tf": "tensorflow", "cv": "computer vision",
    "rdbms": "sql",
}


def _validate_skills(skills: list) -> list[str]:
    if not skills:
        return []
    cleaned, seen = [], set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s = s.strip().strip("\"'.,;:").strip()
        if len(s) < 2 or len(s) > 50:
            continue
        if re.match(r'^[\d\s\W]+$', s):
            continue
        if re.match(r'^[a-zA-Z]$', s):
            continue
        if re.search(r'[{};:@#]', s):
            continue
        s_lower = s.lower()
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


# ── HTML-based multi-job extraction ──────────────────────────────────────────

def _extract_jobs_from_html(email: dict) -> list[dict]:
    """
    Extract ALL jobs from HTML email body.
    Works by finding repeated job-block patterns in HTML.
    This is the primary extractor for emails containing multiple jobs.
    """
    body = email.get("body", "")
    if not body:
        return []

    parsed    = parse_email_html(body)
    text      = parsed["text"]
    all_links = parsed["links"]
    job_links = parsed["job_links"]

    if not text:
        return []

    # Try BeautifulSoup structured extraction
    try:
        jobs = _bs4_multi_job_extract(body, email, all_links, job_links)
        if jobs:
            return jobs
    except Exception as e:
        logger.debug(f"BS4 extraction failed: {type(e).__name__}")

    # Fall back to text-based block splitting
    return _text_block_multi_job_extract(text, email, all_links, job_links)


def _bs4_multi_job_extract(
    html: str, email: dict, all_links: list, job_links: list
) -> list[dict]:
    """
    Extract jobs using BeautifulSoup by finding repeated structural patterns.
    Works well for Glassdoor, Naukri, LinkedIn job alert emails.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    jobs = []

    # Pattern 1: Each job is in a <tr> or <td> block with title + company
    # Pattern 2: Each job is in a <div> with class containing "job"
    # Pattern 3: Jobs separated by horizontal rules

    # Find all anchor tags with job-like text
    apply_links_map = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        if href.startswith(("http://","https://")):
            apply_links_map[text[:80]] = href

    # Try finding table rows — each row often = one job
    rows = soup.find_all("tr")
    if len(rows) >= 3:
        for row in rows:
            job = _parse_job_from_element(row, email, apply_links_map)
            if job:
                jobs.append(job)
        if jobs:
            return _deduplicate_jobs(jobs)

    # Try finding div blocks
    for cls_pattern in ["job","listing","opportunity","position","role","card","item"]:
        blocks = soup.find_all(
            "div",
            class_=re.compile(cls_pattern, re.IGNORECASE)
        )
        if blocks and len(blocks) >= 2:
            for block in blocks:
                job = _parse_job_from_element(block, email, apply_links_map)
                if job:
                    jobs.append(job)
            if jobs:
                return _deduplicate_jobs(jobs)

    return []


def _parse_job_from_element(element, email: dict, links_map: dict) -> dict | None:
    """Parse a single job from an HTML element (tr, div, etc.)."""
    try:
        text = element.get_text(separator=" ", strip=True)
        if len(text) < 10:
            return None

        # Need at least some indication this is a job
        if not re.search(
            r'\b(engineer|developer|analyst|scientist|manager|intern|'
            r'designer|architect|lead|senior|junior|associate|consultant)\b',
            text, re.IGNORECASE
        ):
            return None

        # Extract apply link from this element
        link = None
        for a in element.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("http://","https://")):
                link = href
                break

        return _parse_job_from_text_block(text, email, link)
    except Exception:
        return None


def _text_block_multi_job_extract(
    text: str, email: dict, all_links: list, job_links: list
) -> list[dict]:
    """
    Split text into job blocks and parse each.
    Works when HTML parsing doesn't find structure.
    """
    # Split strategies for multi-job emails
    split_patterns = [
        r'\n\d+[d|h]\s*\n',           # Glassdoor: "2d" separator
        r'\n(?=\d+\.\s+[A-Z])',        # numbered list
        r'\n(?=[A-Z][^a-z]{0,3}\s)',   # new title pattern
        r'[-─—]{3,}',                   # horizontal separator
        r'\n\n(?=[A-Z])',               # double newline before capital
    ]

    blocks = None
    for pattern in split_patterns:
        parts = re.split(pattern, text)
        if len(parts) >= 3:
            blocks = parts
            break

    if not blocks:
        # Last resort: split on job title patterns
        blocks = re.split(
            r'\n(?=(?:Senior|Junior|Lead|Principal|Staff|Associate|Mid|Entry)?\s*'
            r'(?:Software|Data|ML|AI|Backend|Frontend|Full.?Stack|DevOps|'
            r'Cloud|Product|UX|UI|QA|Test|Security)\s+)',
            text
        )

    if not blocks:
        blocks = [text]

    jobs = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if len(block) < 20:
            continue
        link = job_links[i] if i < len(job_links) else (all_links[i] if i < len(all_links) else None)
        job  = _parse_job_from_text_block(block, email, link)
        if job:
            jobs.append(job)

    return _deduplicate_jobs(jobs)


# Compiled patterns for speed
_ROLE_RE = [
    re.compile(
        r'^(.{5,80}?)\s+at\s+(.{3,60}?)(?:\s+and\s+\d+|\s*[-|,].*)?$',
        re.IGNORECASE | re.MULTILINE
    ),
    re.compile(r'([A-Z][A-Za-z\s/&+\-\.]{5,60}?)\s+at\s+([A-Z][A-Za-z\s\.]{3,50})', re.MULTILINE),
    re.compile(r'^([A-Z][A-Za-z\s/&+\-\.]{5,60})\s*[-–|]\s*([A-Z][A-Za-z\s\.]{3,50})', re.MULTILINE),
]

_SALARY_RE = [
    re.compile(
        r'(?:salary|ctc|package|compensation)[:\s]*'
        r'([₹$€]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K|L|per\s+month|pm|pa)?'
        r'(?:\s*[-–]\s*[₹$€]?\s*[\d,\.]+\s*(?:LPA|lpa|lac|lakh|k|K)?)?)',
        re.IGNORECASE
    ),
    re.compile(r'([₹$]\s*[\d,\.]+(?:\s*[-–]\s*[₹$]\s*[\d,\.]+)?\s*(?:LPA|lpa|K|k))', re.IGNORECASE),
    re.compile(r'(\d+[-–]\d+\s*(?:LPA|lpa|lac|lakh))', re.IGNORECASE),
]

_LOCATION_RE = [
    re.compile(r'(?:location|city|place|based\s+in|office)[:\s]+([A-Z][A-Za-z\s,\.]{3,50})', re.IGNORECASE),
    re.compile(r'\b(Bangalore|Mumbai|Delhi|Hyderabad|Chennai|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Kochi|Indore|Bhopal|Lucknow)\b', re.IGNORECASE),
    re.compile(r'\b(remote|work\s+from\s+home|wfh|hybrid|onsite|on-site)\b', re.IGNORECASE),
]

_SKILL_SECTION_RE = re.compile(
    r'(?:required\s+skills?|skill\s+set|tech(?:nical)?\s+skills?|'
    r'requirements?|qualifications?|must\s+have|key\s+skills?)'
    r'[:\s]*([^\n]{10,400})',
    re.IGNORECASE
)


def _extract_skills_from_text(text: str) -> list[str]:
    skills = []
    # Section-based extraction
    for match in _SKILL_SECTION_RE.findall(text):
        parts = re.split(r'[,;•·|/]', match)
        for p in parts:
            p = p.strip()
            if 2 <= len(p) <= 40:
                skills.append(p)
    # Keyword matching
    text_lower = text.lower()
    for skill in _VALID_SKILLS:
        if skill in text_lower:
            idx    = text_lower.find(skill)
            actual = text[idx:idx + len(skill)]
            skills.append(actual)
    return _validate_skills(skills)


def _parse_job_from_text_block(
    text: str, email: dict, link: str | None
) -> dict | None:
    """Parse a single job from a text block."""
    if not text or len(text) < 15:
        return None

    role, company = "", ""

    # Try subject line first
    subject = email.get("subject","")
    for pat in _ROLE_RE:
        m = pat.match(subject.strip())
        if m:
            role    = m.group(1).strip()[:100]
            company = m.group(2).strip()[:100] if m.lastindex >= 2 else ""
            break

    # Try text block
    if not role:
        for pat in _ROLE_RE:
            m = pat.search(text)
            if m:
                r = m.group(1).strip()
                c = m.group(2).strip() if m.lastindex >= 2 else ""
                if len(r) >= 3 and not r.lower().startswith(("http","dear","hi ")):
                    role    = r[:100]
                    company = c[:100]
                    break

    # Must have a role
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
            if len(loc) >= 3:
                location = _normalize_location(loc)
                break

    # Skills
    skills = _extract_skills_from_text(text)

    # Description — first meaningful sentence
    sentences = [s.strip() for s in text.replace("\n"," ").split(".") if len(s.strip()) > 25]
    desc      = (sentences[0] + ".") if sentences else f"{role} at {company}"

    return {
        "role":          role,
        "company":       company or "Unknown",
        "location":      location,
        "salary":        salary,
        "skills":        skills,
        "link":          link,
        "all_links":     [link] if link else [],
        "description":   desc[:200],
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


# ── LLM multi-job extraction (fallback) ──────────────────────────────────────

_LLM_MULTI_JOB_PROMPT = """\
This email contains multiple job listings. Extract ALL of them.

Email subject: {subject}
Email content:
{body}

Extract every distinct job. For EACH job return:
{{"role":"exact title","company":"company name","location":"city or Remote","salary":"range or Not specified","skills":["python","sql"],"description":"1 sentence"}}

RULES:
- skills must be real technologies (python, java, sql, react etc.) NOT single letters
- Extract ALL jobs you can find, even if there are 10+
- Return [] if truly no jobs found

Return JSON array only, no explanation:"""


@lru_cache(maxsize=300)
def _cached_llm(prompt: str) -> str:
    return call_llm(prompt, temperature=0.0, max_tokens=2000, use_cache=True)


def _llm_multi_job_extract(email: dict) -> list[dict]:
    """LLM extraction — finds all jobs including deeply nested ones."""
    body  = email.get("body", "")
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
            links = parse_email_html(body).get("job_links", [])
            jobs  = []
            for i, job in enumerate(data):
                if not isinstance(job, dict):
                    continue
                role = str(job.get("role","")).strip()
                if len(role) < 3:
                    continue
                skills = _validate_skills(job.get("skills",[]))
                jobs.append({
                    "role":          role[:100],
                    "company":       str(job.get("company","Unknown")).strip()[:100],
                    "location":      _normalize_location(job.get("location","")),
                    "salary":        _normalize_salary(job.get("salary","")),
                    "skills":        skills,
                    "link":          links[i] if i < len(links) else None,
                    "all_links":     [links[i]] if i < len(links) else [],
                    "description":   str(job.get("description","")).strip()[:200],
                    "email_id":      email.get("id",""),
                    "email_subject": email.get("subject",""),
                    "source":        "llm",
                })
            logger.info(f"LLM extracted {len(jobs)} jobs from email")
            return jobs
    except Exception as e:
        logger.warning(f"LLM multi-job extract failed: {type(e).__name__}")

    return []


def extract_jobs_from_email(email: dict) -> list[dict]:
    """
    Extract ALL jobs from a single email.
    Strategy:
    1. HTML-based block detection (finds repeating job patterns)
    2. LLM fallback for unstructured emails
    Rate limited: 0.2s between calls.
    """
    # Step 1: HTML-aware extraction
    jobs = _extract_jobs_from_html(email)
    logger.info(f"HTML extraction found {len(jobs)} jobs from email {email.get('id','')[:20]}")

    if not jobs:
        # Step 2: LLM fallback
        time.sleep(0.2)
        jobs = _llm_multi_job_extract(email)
        logger.info(f"LLM extraction found {len(jobs)} jobs from email {email.get('id','')[:20]}")

    return jobs


def process_job_emails(
    emails: list[dict],
    enrich: bool = False,
    max_emails: int = 30,
) -> list[dict]:
    """Process multiple job emails, extract all jobs, deduplicate."""
    all_jobs = []
    for email in emails[:max_emails]:
        try:
            jobs = extract_jobs_from_email(email)
            if enrich:
                jobs = [_enrich_if_safe(j) for j in jobs]
                time.sleep(0.3)
            all_jobs.extend(jobs)
        except Exception as e:
            logger.warning(f"Job extraction error: {type(e).__name__}")

    # Final deduplication: (role, company, location)
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

    logger.info(f"Total unique jobs from {len(emails)} emails: {len(unique)}")
    return unique


def _should_skip_scrape(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(blocked in domain for blocked in _NO_SCRAPE)
    except Exception:
        return True


def _enrich_if_safe(job: dict) -> dict:
    link = job.get("link")
    if not link or _should_skip_scrape(link):
        return job
    try:
        import requests
        resp = requests.get(
            link,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        resp.raise_for_status()
        extra_text  = clean_email_body(resp.text, max_chars=2000)
        extra_skills = _extract_skills_from_text(extra_text)
        if extra_skills:
            merged        = list(dict.fromkeys((job.get("skills") or []) + extra_skills))[:25]
            job["skills"] = _validate_skills(merged)
            job["scraped"] = True
    except Exception:
        pass
    return job


# ── Resume parsing ─────────────────────────────────────────────────────────────

_RESUME_PARSE_PROMPT = """\
Parse this resume carefully. Return JSON only.

{resume_text}

Return this exact structure:
{{
  "name": "full name",
  "skills": ["python", "machine learning", "sql"],
  "experience_years": 1,
  "experience": "2-3 sentence summary of work experience",
  "current_role": "most recent job title",
  "education": "degree and institution",
  "projects": ["project name and brief description"],
  "certifications": ["certification name"],
  "summary": "2-3 sentence professional summary"
}}

IMPORTANT:
- skills must be real technologies, minimum 3 characters
- experience_years should be realistic based on dates found
- name must be the full name, not truncated"""


def _clean_resume_text(raw: str) -> str:
    """
    Clean PDF-extracted text:
    1. Fix line breaks within words
    2. Normalize whitespace
    3. Preserve section headers
    """
    # Fix common PDF extraction issues
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Fix broken words (word split across lines without hyphen)
    lines = text.split("\n")
    fixed = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            fixed.append("")
            continue
        # If line ends mid-word and next line starts with lowercase, join
        if (i + 1 < len(lines) and
            line and
            line[-1].isalpha() and
            lines[i+1].strip() and
            lines[i+1].strip()[0].islower()):
            fixed.append(line + " " + lines[i+1].strip())
            lines[i+1] = ""
        else:
            fixed.append(line)

    text = "\n".join(fixed)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Fix common OCR/PDF broken word patterns
    text = re.sub(r'(\w{3,})\s+(ously|tion|ing|ment|ness|ity|ance|ence|ible|able)\b',
                  r'\1\2', text)

    return text.strip()


def _extract_sections(text: str) -> dict:
    """
    Split resume into sections for better structured parsing.
    Returns {section_name: content}
    """
    section_headers = [
        "professional summary", "summary", "objective",
        "education", "academic background",
        "experience", "work experience", "internship", "internships",
        "technical skills", "skills", "core competencies",
        "projects", "project",
        "certifications", "achievements", "awards",
        "languages", "interests",
    ]

    pattern = re.compile(
        r'(?:^|\n)(' +
        "|".join(re.escape(h) for h in section_headers) +
        r')[\s:]*\n',
        re.IGNORECASE
    )

    sections    = {}
    last_header = "intro"
    last_pos    = 0

    for m in pattern.finditer(text):
        header = m.group(1).lower().strip()
        if last_pos < m.start():
            sections[last_header] = text[last_pos:m.start()].strip()
        last_header = header
        last_pos    = m.end()

    sections[last_header] = text[last_pos:].strip()
    return sections


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""

    # Try pdfminer first (best quality)
    try:
        import pdfminer.high_level as pmh
        import io
        text = pmh.extract_text(io.BytesIO(file_bytes)) or ""
    except Exception:
        pass

    # Try pypdf
    if len(text.strip()) < 100:
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            pass

    # Raw decode fallback
    if len(text.strip()) < 100:
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)

    return _clean_resume_text(text)


def parse_resume(file_bytes: bytes, file_type: str = "pdf") -> dict:
    """
    Parse resume using section detection + LLM.
    Returns structured dict with name, skills, experience etc.
    """
    if file_type == "pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = file_bytes.decode("utf-8", errors="ignore")
        raw_text = _clean_resume_text(raw_text)

    raw_text = raw_text.strip()

    if len(raw_text) < 100:
        logger.warning(f"Resume text too short ({len(raw_text)} chars)")
        return _empty_resume()

    # Extract skills with rule-based first
    rule_skills = _extract_skills_from_text(raw_text)

    # Use LLM for full structured parsing
    try:
        # Feed sections if detected
        sections = _extract_sections(raw_text)
        context  = raw_text

        if len(sections) > 2:
            # Build focused context from sections
            context_parts = []
            for sec_name, sec_content in sections.items():
                if sec_content and len(sec_content) > 10:
                    context_parts.append(f"=== {sec_name.upper()} ===\n{sec_content}")
            context = "\n\n".join(context_parts)

        prompt = _RESUME_PARSE_PROMPT.format(resume_text=context[:3500])
        raw    = _cached_llm(prompt)
        m      = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data   = json.loads(m.group(0))
            llm_sk = _validate_skills(data.get("skills", []))
            merged = list(dict.fromkeys(llm_sk + rule_skills))[:30]

            # Clean up name — sometimes truncated
            name = str(data.get("name","")).strip()
            if len(name) < 2:
                # Try to find name from raw text
                first_line = raw_text.split("\n")[0].strip()
                if len(first_line) > 2 and len(first_line) < 50:
                    name = first_line

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

    # Fallback: rule-based only
    first_line = raw_text.split("\n")[0].strip()
    name       = first_line if (2 < len(first_line) < 50) else ""

    return {
        "name":            name,
        "skills":          rule_skills,
        "experience_years": 0,
        "experience":      "",
        "current_role":    "",
        "education":       "",
        "projects":        [],
        "certifications":  [],
        "summary":         f"Resume parsed with {len(rule_skills)} skills detected",
        "raw_text_length": len(raw_text),
    }


def _empty_resume() -> dict:
    return {
        "name":"","skills":[],"experience_years":0,"experience":"",
        "current_role":"","education":"","projects":[],"certifications":[],
        "summary":"","raw_text_length":0,
    }


# ── Job-Resume scoring (fixed 100% bug) ──────────────────────────────────────

_MATCH_PROMPT = """\
Score this job-resume match. Return JSON only.

Job: {role} at {company}
Required skills: {job_skills}
Job description: {description}

Candidate profile:
Skills: {resume_skills}
Experience: {experience_years} years
Current role: {current_role}
Education: {education}

Scoring (be realistic, NOT inflated):
- Skill overlap: 50 points max
- Role relevance: 30 points max
- Experience fit: 20 points max

{{
  "match_score": 65,
  "matched_skills": ["python","sql"],
  "missing_skills": ["java","kubernetes"],
  "strengths": "specific strengths",
  "gaps": "what is missing",
  "recommendation": "Good Match",
  "fit_reason": "You match X and Y strongly but lack Z",
  "ready_to_apply": true,
  "skill_gap_suggestions": [
    {{"skill": "kubernetes", "resource": "Kubernetes on Coursera"}}
  ]
}}

IMPORTANT: match_score must be 0-97 max. Never give 100 unless perfect match on every criterion."""


def score_job_match(job: dict, resume: dict) -> dict:
    job_skills    = {s.lower() for s in _validate_skills(job.get("skills",[]))}
    resume_skills = {s.lower() for s in _validate_skills(resume.get("skills",[]))}

    if not job_skills:
        rule_score = 45
        matched, missing = [], []
    else:
        matched_set = job_skills & resume_skills
        missing_set = job_skills - resume_skills
        matched     = sorted(matched_set)
        missing     = sorted(missing_set)
        overlap     = (len(matched_set) / len(job_skills)) * 50
        role_score  = _role_relevance(job, resume) * 30
        base        = 20
        rule_score  = min(97, int(overlap + role_score + base))

    def _rec(s):
        if s >= 80: return "Strong Match"
        if s >= 60: return "Good Match"
        if s >= 40: return "Partial Match"
        return "Weak Match"

    if job_skills and resume_skills:
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
                    "ready_to_apply":        bool(data.get("ready_to_apply", score >= 60)),
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
        "ready_to_apply":        rule_score >= 60,
        "skill_gap_suggestions": [
            {"skill": s, "resource": f"Search '{s}' on Coursera or Udemy"}
            for s in list(missing)[:3]
        ],
    }


def _role_relevance(job: dict, resume: dict) -> float:
    job_role    = job.get("role","").lower()
    resume_role = resume.get("current_role","").lower()
    if not job_role or not resume_role:
        return 0.5
    jt = set(re.findall(r'\b\w{3,}\b', job_role))
    rt = set(re.findall(r'\b\w{3,}\b', resume_role))
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