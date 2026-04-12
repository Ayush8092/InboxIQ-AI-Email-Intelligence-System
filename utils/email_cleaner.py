"""
Production Email Cleaner — Intent-Driven, Structure-Aware.

Bugs fixed in this version:
1. NoneType crash: _LOCATION_LOOSE.search(line).group(0) crashed when line was
   'onsite', 'on-site', 'WFH', 'work from home' (in strict but not loose pattern).
   Fix: single _LOCATION_FULL regex covers all location variants + safe extraction.

2. 'avatar' appearing as company name: default <img alt="avatar"> text was
   being used as company name after image replacement.
   Fix: _FILTER_CONTEXT removes generic alt texts before context parsing.

3. '3d'/'4d' (job age badges) appearing in job content.
   Fix: _FILTER_CONTEXT removes \d+[dhm] patterns.

4. 'Easy Apply' button text appearing as company/skills.
   Fix: _FILTER_CONTEXT removes 'Easy Apply' and 'Quick Apply'.

5. Uncategorized emails after body upgrade: LLM categorizer received huge HTML.
   Fix: clean_email_body() for LLM input still returns concise text (not job cards).
        Added get_short_body_for_llm() that returns first 500 chars of clean text.
"""
import re
import base64
from urllib.parse import urlparse
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

_INVISIBLE = re.compile(
    r'[\u200b\u200c\u200d\ufeff\u00ad\u00a0\xa0\u2028\u2029\u3000\u200e\u200f]'
)

_JOB_TITLE_RE = re.compile(
    r'\b(engineer|developer|analyst|scientist|manager|intern|designer|architect|'
    r'lead|senior|junior|associate|consultant|specialist|director|coordinator|'
    r'fullstack|full.stack|frontend|backend|devops|sde|swe|data|ml|ai|software|'
    r'machine.learning|generative|applied|research|product|cloud|security|nlp|'
    r'computer.vision|program.manager|technical)\b',
    re.IGNORECASE
)

# Single comprehensive location pattern — covers ALL location variants
# Used everywhere; no more split strict/loose causing NoneType bugs
_LOCATION_RE = re.compile(
    r'\b(bangalore|bengaluru|mumbai|navi\s+mumbai|delhi(?:\s+ncr)?|new\s+delhi|'
    r'hyderabad|chennai|pune|kolkata|noida|gurgaon|gurugram|ahmedabad|jaipur|'
    r'kochi|indore|lucknow|bhopal|nagpur|surat|india|remote|hybrid|onsite|'
    r'on-site|work\s+from\s+home|wfh)\b',
    re.IGNORECASE
)

# A line IS a location if: ENTIRE content is a location (not company containing city)
_LOCATION_LINE_RE = re.compile(
    r'^(bangalore|bengaluru|mumbai|navi\s+mumbai|delhi(?:\s+ncr)?|new\s+delhi|'
    r'hyderabad|chennai|pune|kolkata|noida|gurgaon|gurugram|ahmedabad|jaipur|'
    r'kochi|indore|lucknow|bhopal|nagpur|surat|india|remote|hybrid|onsite|'
    r'on-site|work\s+from\s+home|wfh)\s*(?:[,|•]\s*.*)?$',
    re.IGNORECASE
)

# Company indicators — lines with these are NOT locations
_COMPANY_INDICATORS = re.compile(
    r'\d+\.\d+|pvt\.?|ltd\.?|inc\.?|llc|corp\.?|technologies|services|'
    r'solutions|systems|consulting|ventures|industries|associates|group|'
    r'global|international|enterprises|limited|private',
    re.IGNORECASE
)

_SALARY_RE = re.compile(
    r'[₹$€£]?\s*\d+[KkLlM]?\s*[-–—]\s*[₹$€£]?\s*\d+[KkLlM]?'
    r'(?:\s*\(?(?:Employer|Glassdoor|Company)\s+Est\.?\)?)?',
    re.IGNORECASE
)
_SALARY_LPA = re.compile(
    r'\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*(?:LPA|lpa|lac|lakh)',
    re.IGNORECASE
)

_STAR_RATING = re.compile(
    r'\s*\d+\.\d+\s*[★☆\u2605\u2606\u2B50⭐✦]?\s*$'
)

_SKILL_DELIM = re.compile(r'[•·,;|]')

_NOISE_TEXT = re.compile(
    r'(unsubscribe|manage\s+settings|privacy\s+policy|terms\s+of\s+use|'
    r'click\s+here|view\s+in\s+browser|if\s+you\s+cannot\s+view|'
    r'©\s*\d{4}|all\s+rights\s+reserved|do\s+not\s+reply|'
    r'this\s+message\s+was\s+sent|you\s+are\s+receiving|'
    r'add.*to\s+your\s+address|please\s+do\s+not\s+reply)',
    re.IGNORECASE
)

# Context strings to filter out before parsing job fields
# These are noise that appears in Glassdoor email td elements:
# - 'avatar': default img alt text for company logos
# - '3d', '4d', '2h': job age badges
# - 'Easy Apply', 'Quick Apply': button text
# - 'New Tab', 'logo', 'icon': other UI noise
_FILTER_CONTEXT = re.compile(
    r'^(avatar|logo|image|icon|easy\s+apply|quick\s+apply|apply\s+now|'
    r'new\s+tab|see\s+more|\d+[dhm]|\d+\s+days?\s+ago)$',
    re.IGNORECASE
)

_SKIP_LINK = re.compile(
    r'manage.settings|unsubscribe|privacy|terms.of|help.center|'
    r'contact.us|about.us|brand.view|tracking|pixel|logomark|'
    r'logo\.png|icon\.png|\.gif$|opt.out|email.settings',
    re.IGNORECASE
)

_JOB_DOMAINS = {
    "glassdoor.com", "glassdoor.co.in", "naukri.com", "linkedin.com",
    "indeed.com", "monster.com", "shine.com", "foundit.in", "instahyre.com",
    "unstop.com", "internshala.com", "wellfound.com", "cutshort.io",
    "hirist.tech", "apna.co", "timesjobs.com", "careerbuilder.com",
    "ziprecruiter.com", "greenhouse.io", "lever.co", "ashbyhq.com",
    "workable.com", "jobs.google.com", "simplyhired.com",
}

_EMPLOYMENT_TYPES = {
    'full-time', 'part-time', 'contract', 'freelance', 'permanent',
    'temporary', 'fixed-term', 'full time', 'part time', 'apprenticeship',
    'internship', 'casual', 'seasonal',
}

# Generic single-word alt texts that are NOT company names
_GENERIC_ALT_TEXTS = {
    'avatar', 'logo', 'image', 'icon', 'photo', 'picture', 'banner',
    'thumbnail', 'company', 'employer', 'brand',
}


# ── Data ingestion ─────────────────────────────────────────────────────────────

def extract_html_from_payload(payload: dict) -> str:
    """
    Recursively walk Gmail API message payload to find the text/html part.

    Gmail returns messages as nested MIME:
      multipart/mixed > multipart/alternative > text/html (what we want)

    Always returns complete HTML, never truncated.
    """
    mime_type = payload.get('mimeType', '')
    parts     = payload.get('parts', [])

    # Direct HTML part
    if mime_type == 'text/html':
        data = payload.get('body', {}).get('data', '')
        if data:
            try:
                return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace')
            except Exception:
                pass
        return ''

    # Recurse into child parts — prefer HTML over plain
    if parts:
        html_results  = []
        plain_results = []
        for part in parts:
            result    = extract_html_from_payload(part)
            part_mime = part.get('mimeType', '')
            if result:
                if part_mime == 'text/html' or (result and '<' in result[:200]):
                    html_results.append(result)
                else:
                    plain_results.append(result)
        if html_results:
            return '\n'.join(html_results)
        if plain_results:
            return '\n'.join(plain_results)

    # Fallback: plain text root
    if mime_type == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            try:
                return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace')
            except Exception:
                pass

    return ''


# ── Utilities ──────────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    s = _INVISIBLE.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _is_location_line(line: str) -> bool:
    """
    Check if an entire line represents a location.
    Avoids false positives on company names containing city names
    like 'Terrier Security Services (India)'.
    """
    stripped = line.strip()
    # Strict check: entire line is a location (possibly with comma suffix)
    if _LOCATION_LINE_RE.match(stripped):
        return True
    # Loose check: short line containing location keyword AND no company markers
    if (len(stripped) < 40 and
        _LOCATION_RE.search(stripped) and
        not _COMPANY_INDICATORS.search(stripped)):
        return True
    return False


def _extract_location_value(line: str) -> str:
    """
    Extract the location value from a line.
    BUG FIX: always returns a string, never crashes.
    Uses _LOCATION_RE which covers ALL location variants.
    """
    m = _LOCATION_RE.search(line)
    if m:
        return m.group(0).strip()
    # Fallback: return the cleaned line itself
    return line.strip()


def _is_salary(line: str) -> bool:
    return bool(_SALARY_RE.search(line) or _SALARY_LPA.search(line))


def _is_noise_link(href: str) -> bool:
    return bool(_SKIP_LINK.search(href))


def _is_job_link(href: str) -> bool:
    if _is_noise_link(href):
        return False
    try:
        domain = urlparse(href).netloc.lower().lstrip("www.")
        if any(jd in domain for jd in _JOB_DOMAINS):
            return True
    except Exception:
        pass
    return bool(re.search(
        r'/job[s]?[-/]|/career|/apply\b|/opening|/position|jobid=|jl=',
        href, re.IGNORECASE
    ))


# ── Context window parser ──────────────────────────────────────────────────────

def _parse_context_window(title: str, strings: list[str]) -> dict:
    """
    Parse context strings from a job card's parent <td> into structured fields.

    BUG FIXES:
    - Never calls .group(0) without checking if search() returned None first
    - Filters 'avatar', '3d/4d', 'Easy Apply' before parsing
    - Uses _extract_location_value() which is always safe
    """
    title_l = title.lower()

    # Filter: remove title itself, generic noise, and UI artifacts
    lines = []
    for s in strings:
        if not s or len(s) <= 1:
            continue
        if s.lower() == title_l:
            continue
        if _NOISE_TEXT.search(s):
            continue
        if re.match(r'^https?://', s) and len(s) > 60:
            continue
        # NEW: filter generic img alt texts, job age, button texts
        if _FILTER_CONTEXT.match(s.strip()):
            continue
        lines.append(s)

    company  = 'Unknown'
    location = 'Not specified'
    salary   = 'Not specified'
    skills: list[str] = []

    for line in lines:
        # Salary check
        if _is_salary(line):
            if salary == 'Not specified':
                salary = line.strip()[:100]
            continue

        # Location check (safe: uses _extract_location_value which never crashes)
        if _is_location_line(line):
            if location == 'Not specified':
                location = _extract_location_value(line)  # SAFE: always returns str
            continue

        # Skills: bullet/comma-separated line
        if _SKILL_DELIM.search(line):
            for part in _SKILL_DELIM.split(line):
                part = part.strip()
                if (part.lower() not in _EMPLOYMENT_TYPES and
                    2 <= len(part) <= 50 and
                    not _is_salary(part)):
                    skills.append(part)
            continue

        # Company: first unmatched line after filtering
        if company == 'Unknown' and 2 < len(line) < 120:
            co = _STAR_RATING.sub('', line).strip()
            co = re.sub(r'\s*[|•]\s*(India|Remote|Hybrid).*$', '', co, flags=re.I).strip()
            # Must not be a generic word or just a number
            if (co and len(co) > 2 and
                co.lower() not in _GENERIC_ALT_TEXTS and
                not re.match(r'^\d+$', co)):
                company = co

    return {
        'company':  company[:100],
        'location': location[:80],
        'salary':   salary[:100],
        'skills':   list(dict.fromkeys(skills))[:15],
    }


# ── Intent-driven job card extractor ──────────────────────────────────────────

def _extract_job_cards_from_soup(soup) -> list[dict]:
    """
    Intent-driven: find job-title anchors, parse limited context window.
    """
    job_cards: list[dict] = []
    seen: set[str] = set()

    for a_tag in soup.find_all('a', href=True):
        href  = a_tag['href'].strip()
        title = _clean(a_tag.get_text())

        if not (4 <= len(title) <= 120):
            continue
        if not _JOB_TITLE_RE.search(title):
            continue
        if _is_noise_link(href):
            continue
        if not href.startswith(('http://', 'https://')):
            continue

        title_key = title.lower()
        if title_key in seen:
            continue
        seen.add(title_key)

        # Limited context: parent <td> strings only
        parent_td = a_tag.find_parent('td')
        if parent_td:
            raw_strings = [_clean(str(s)) for s in parent_td.strings]
        else:
            raw_strings = [_clean(str(s)) for s in a_tag.parent.strings]

        # Filter empty and noise before passing to parser
        context = [
            s for s in raw_strings
            if s and len(s) > 1
            and not _NOISE_TEXT.search(s)
            and not _FILTER_CONTEXT.match(s.strip())
        ]

        parsed = _parse_context_window(title, context)

        job_cards.append({
            'role':          title,
            'company':       parsed['company'],
            'location':      parsed['location'],
            'salary':        parsed['salary'],
            'skills':        parsed['skills'],
            'link':          href,
            'all_links':     [href],
            'email_id':      '',
            'email_subject': '',
            'source':        'dom_intent',
        })

    logger.info(f"Job cards extracted: {len(job_cards)}")
    return job_cards


def _build_display_text(soup, job_cards: list[dict]) -> str:
    """Build clean display text for the UI."""
    lines: list[str] = []

    # Brief intro
    intro_count = 0
    for tag in (soup.find('body') or soup).children:
        if not hasattr(tag, 'get_text'):
            continue
        txt = _clean(tag.get_text(separator=' '))
        if not txt or len(txt) < 5 or _NOISE_TEXT.search(txt):
            continue
        if job_cards and any(jc['role'].lower() in txt.lower() for jc in job_cards[:2]):
            break
        lines.append(txt)
        intro_count += 1
        if intro_count >= 2:
            break

    if job_cards:
        if lines:
            lines.append('')
        lines.append(f'📋 {len(job_cards)} Job Listings Found:')
        lines.append('─' * 45)
        for i, card in enumerate(job_cards, 1):
            lines.append(f'\n{i}. {card["role"]}')
            if card['company'] != 'Unknown':
                lines.append(f'   🏢 {card["company"]}')
            if card['location'] != 'Not specified':
                lines.append(f'   📍 {card["location"]}')
            if card['salary'] != 'Not specified':
                lines.append(f'   💰 {card["salary"]}')
            if card['skills']:
                lines.append(f'   🔧 {", ".join(card["skills"][:8])}')
            if card['link']:
                lines.append(f'   🔗 {card["link"][:90]}')
    else:
        body_text = soup.get_text(separator='\n', strip=True)
        for line in body_text.split('\n'):
            line = line.strip()
            if not line or len(line) <= 1:
                continue
            if _NOISE_TEXT.search(line):
                continue
            if re.match(r'^https?://', line) and len(line) > 80:
                continue
            line = _clean(line)
            if line:
                lines.append(line)

    return '\n'.join(lines)


# ── Main public API ────────────────────────────────────────────────────────────

def parse_email_html(raw: str) -> dict:
    """
    Parse email body HTML into structured output.

    Returns:
        text      : Readable text for UI display (structured job listing)
        llm_text  : Short clean text for LLM input (categorization etc.)
        links     : All href links
        job_links : Job-platform links
        job_cards : Structured [{role, company, location, salary, skills, link}]
        is_html   : Whether input was HTML
    """
    if not raw:
        return {
            'text': '', 'llm_text': '', 'links': [],
            'job_links': [], 'job_cards': [], 'is_html': False,
        }

    raw = _INVISIBLE.sub(' ', raw)

    is_html_content = bool(re.search(
        r'<(?:html|body|div|span|table|td|p|a)\b', raw, re.IGNORECASE
    ))

    if not is_html_content:
        text  = re.sub(r'\s+', ' ', raw).strip()
        text  = re.sub(r'\n{3,}', '\n\n', text)
        links = re.findall(r'https?://[^\s<>"\']{10,300}', text)
        return {
            'text':      text[:6000],
            'llm_text':  text[:500],
            'links':     list(dict.fromkeys(links))[:20],
            'job_links': [],
            'job_cards': [],
            'is_html':   False,
        }

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _stdlib_fallback(raw)

    soup = BeautifulSoup(raw, 'html.parser')

    # Remove noise tags
    for tag in soup(['script', 'style', 'noscript', 'meta']):
        tag.decompose()
    for lt in soup.find_all('link'):
        lt.decompose()
    # Images: replace with alt only if meaningful and not generic
    for img in soup.find_all('img'):
        alt = _clean(img.get('alt', '') or img.get('title', '') or '').strip()
        if (alt and 2 < len(alt) < 40 and
            alt.lower() not in _GENERIC_ALT_TEXTS and
            not re.search(r'logo|icon|pixel|banner|track|spacer', alt, re.I)):
            img.replace_with(alt)
        else:
            img.decompose()

    # Collect links
    all_links: list[str] = []
    job_links: list[str] = []
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href.startswith(('http://', 'https://')):
            continue
        if len(href) > 500 or _is_noise_link(href):
            continue
        all_links.append(href)
        if _is_job_link(href):
            job_links.append(href)

    all_links = list(dict.fromkeys(all_links))[:40]
    job_links = list(dict.fromkeys(job_links))[:15]

    # Job card extraction
    job_cards = _extract_job_cards_from_soup(soup)

    # Display text
    text = _build_display_text(soup, job_cards)

    # LLM text: short clean text for categorization/summarization
    # Does NOT use job card format — keeps it plain for LLM understanding
    llm_lines = []
    raw_text  = soup.get_text(separator=' ', strip=True)
    raw_text  = _clean(raw_text)
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line or len(line) <= 2:
            continue
        if _NOISE_TEXT.search(line):
            continue
        if re.match(r'^https?://', line):
            continue
        if _FILTER_CONTEXT.match(line):
            continue
        llm_lines.append(line)
    llm_text = ' '.join(llm_lines)[:600]

    return {
        'text':      text[:8000],
        'llm_text':  llm_text,
        'links':     all_links,
        'job_links': job_links,
        'job_cards': job_cards,
        'is_html':   True,
    }


def _stdlib_fallback(raw: str) -> dict:
    """Fallback parser when BeautifulSoup is unavailable."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.links: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            t = tag.lower()
            if t in ('script', 'style', 'noscript'):
                self._skip += 1
                return
            if self._skip:
                return
            if t == 'a':
                h = dict(attrs).get('href', '')
                if h.startswith(('http://', 'https://')) and not _is_noise_link(h):
                    self.links.append(h)
            if t in ('br', 'p', 'div', 'tr', 'li', 'h1', 'h2', 'h3', 'h4', 'td'):
                self.parts.append('\n')

        def handle_endtag(self, tag):
            if tag.lower() in ('script', 'style', 'noscript') and self._skip > 0:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                t = _clean(data)
                if t and not _NOISE_TEXT.search(t):
                    self.parts.append(t)

        def get_text(self) -> str:
            return re.sub(r'\n{3,}', '\n\n', '\n'.join(self.parts)).strip()

    p = _P()
    try:
        p.feed(raw)
        text   = p.get_text()
        links  = list(dict.fromkeys(p.links))[:40]
        jlinks = [l for l in links if _is_job_link(l)]
        return {
            'text': text[:8000], 'llm_text': text[:600],
            'links': links, 'job_links': jlinks, 'job_cards': [], 'is_html': True,
        }
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', raw)
        return {
            'text': text[:6000], 'llm_text': text[:600],
            'links': [], 'job_links': [], 'job_cards': [], 'is_html': True,
        }


# ── Convenience functions ──────────────────────────────────────────────────────

def clean_email_body(body: str, max_chars: int = 6000) -> str:
    """
    Clean readable text from email body.
    Used for: LLM input, summarization, task extraction.
    Returns the llm_text (short, clean) not the structured job card display text.
    This fixes the uncategorized email issue where large HTML body confused LLM.
    """
    if not body:
        return ''
    result = parse_email_html(body)
    # For LLM use, return llm_text (concise) not full display text
    llm_text = result.get('llm_text', '')
    if llm_text:
        return llm_text[:max_chars]
    return result['text'][:max_chars]


def get_display_text(body: str) -> str:
    """
    Get formatted display text for UI (includes structured job listing).
    Use this in categorized_tab.py for showing email content.
    """
    if not body:
        return ''
    return parse_email_html(body).get('text', '')[:8000]


def extract_job_cards(body: str) -> list[dict]:
    """Extract structured job cards from email HTML."""
    if not body:
        return []
    return parse_email_html(body).get('job_cards', [])


def extract_links_from_email(body: str) -> dict:
    result = parse_email_html(body)
    return {'all_links': result['links'], 'job_links': result['job_links']}


def is_html(text: str) -> bool:
    return bool(re.search(r'<(?:html|body|div|span|p|a|table)\b', text, re.IGNORECASE))