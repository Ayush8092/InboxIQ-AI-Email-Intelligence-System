"""
Production Email Cleaner — Intent-Driven, Structure-Aware.

Architecture (as described in the analysis):
  OLD (broken): HTML → soup.get_text() → noisy flat text → broken regex
  NEW (correct): HTML → noise removal → anchor-based intent detection
                 → limited context window per job card → structured JSON

Key fixes:
1. Data ingestion: extract_html_from_payload() walks Gmail payload recursively
   to find the ACTUAL text/html part (not just root body.data)
2. Noise removal: remove script/style/noscript/link/img BEFORE parsing
3. Intent anchors: find <a> tags with job titles (high-confidence signals)
4. Limited context window: use parent <td>.strings ONLY (not entire ancestor tree)
5. Fixed company/location bug: strict city regex prevents company names
   containing "(India)" from being parsed as locations
6. Invisible unicode cleaned before parsing
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

# Strict location: line IS a location (not contains location as part of company name)
_LOCATION_STRICT = re.compile(
    r'^(bangalore|bengaluru|mumbai|delhi(?:\s+ncr)?|hyderabad|chennai|pune|kolkata|'
    r'noida|gurgaon|gurugram|ahmedabad|jaipur|kochi|indore|lucknow|bhopal|nagpur|'
    r'surat|india|remote|work\s+from\s+home|wfh|onsite|on-site|hybrid)\s*'
    r'(?:[,\|•]\s*(?:full.time|part.time|contract|remote|india))?$',
    re.IGNORECASE
)
# Loose location: short line containing a city
_LOCATION_LOOSE = re.compile(
    r'\b(bangalore|bengaluru|mumbai|delhi|hyderabad|chennai|pune|kolkata|'
    r'noida|gurgaon|gurugram|india|remote|hybrid)\b',
    re.IGNORECASE
)
# Indicators that a line is a company name (prevents misclassification)
_COMPANY_INDICATORS = re.compile(
    r'\d+\.\d+|pvt|ltd|inc|llc|corp|technologies|services|solutions|systems|'
    r'consulting|ventures|industries|associates|group|global|international',
    re.IGNORECASE
)

_SALARY_RE = re.compile(
    r'[₹$€£]?\s*\d+[KkLlM]?\s*[-–—]\s*[₹$€£]?\s*\d+[KkLlM]?'
    r'(?:\s*\(?(?:Employer|Glassdoor|Company)\s+Est\.?\)?)?',
    re.IGNORECASE
)
# Also match "X LPA" format
_SALARY_LPA = re.compile(r'\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*(?:LPA|lpa|lac|lakh)', re.IGNORECASE)

_STAR_RATING = re.compile(r'\s*\d+\.\d+\s*[★☆\u2605\u2606\u2B50⭐✦]?\s*$')

_SKILL_DELIM = re.compile(r'[•·,;|]')

_NOISE_TEXT = re.compile(
    r'(unsubscribe|manage\s+settings|privacy\s+policy|terms\s+of\s+use|'
    r'click\s+here|view\s+in\s+browser|if\s+you\s+cannot\s+view|'
    r'©\s*\d{4}|all\s+rights\s+reserved|do\s+not\s+reply|'
    r'this\s+message\s+was\s+sent|you\s+are\s+receiving|'
    r'add.*to\s+your\s+address|please\s+do\s+not\s+reply)',
    re.IGNORECASE
)

_SKIP_LINK = re.compile(
    r'manage.settings|unsubscribe|privacy|terms.of|help.center|'
    r'contact.us|about.us|brand.view|tracking|pixel|logomark|'
    r'logo\.png|icon\.png|gif$',
    re.IGNORECASE
)

_JOB_DOMAINS = {
    "glassdoor.com","glassdoor.co.in","naukri.com","linkedin.com","indeed.com",
    "monster.com","shine.com","foundit.in","instahyre.com","unstop.com",
    "internshala.com","wellfound.com","cutshort.io","hirist.tech","apna.co",
    "timesjobs.com","careerbuilder.com","ziprecruiter.com","greenhouse.io",
    "lever.co","ashbyhq.com","workable.com","jobs.google.com","simplyhired.com",
}

_EMPLOYMENT_TYPES = {
    'full-time','part-time','contract','freelance','permanent','temporary',
    'fixed-term','full time','part time','apprenticeship','internship',
    'casual','seasonal',
}


# ── Data ingestion: Gmail payload extraction ───────────────────────────────────

def extract_html_from_payload(payload: dict) -> str:
    """
    Recursively walk Gmail API message payload to find the text/html part.

    Gmail returns messages as nested MIME structures:
      multipart/mixed
        multipart/alternative
          text/plain   <- plain text version
          text/html    <- HTML version (what we want)
        image/png      <- inline images

    This function handles all nesting depths and always returns
    the complete, undecoded HTML body.

    Args:
        payload: Gmail API message['payload'] dict

    Returns:
        Full HTML string, or empty string if not found.
    """
    mime_type = payload.get('mimeType', '')
    parts     = payload.get('parts', [])

    # This part IS the HTML — decode and return it
    if mime_type == 'text/html':
        data = payload.get('body', {}).get('data', '')
        if data:
            try:
                return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Base64 decode failed: {type(e).__name__}")
        return ''

    # Multipart — recurse into child parts, preferring HTML over plain
    if parts:
        html_parts  = []
        plain_parts = []

        for part in parts:
            result = extract_html_from_payload(part)
            if result:
                part_mime = part.get('mimeType', '')
                if part_mime == 'text/html' or '<html' in result.lower()[:100]:
                    html_parts.append(result)
                else:
                    plain_parts.append(result)

        if html_parts:
            return '\n'.join(html_parts)
        if plain_parts:
            return '\n'.join(plain_parts)

    # Fallback: root body has data (single-part message)
    if mime_type == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            try:
                return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace')
            except Exception:
                pass

    return ''


# ── Text utilities ─────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    """Remove invisible unicode and normalize whitespace."""
    s = _INVISIBLE.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _is_location(line: str) -> bool:
    """
    Check if a line represents a location.
    Uses strict match to avoid misclassifying company names like
    'Terrier Security Services (India)' as locations.
    """
    stripped = line.strip()
    # Strict: the ENTIRE line is a location
    if _LOCATION_STRICT.match(stripped):
        return True
    # Loose: short line with city AND no company indicators
    if (len(stripped) < 40 and
        _LOCATION_LOOSE.search(stripped) and
        not _COMPANY_INDICATORS.search(stripped)):
        return True
    return False


def _is_salary(line: str) -> bool:
    """Check if line contains salary information."""
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
    Parse an ordered list of text strings from a job card's limited context window.

    These strings come from a single parent <td>, in DOM order:
      [title, company+rating, location, salary, skills•tags, ...]

    Returns structured dict with role/company/location/salary/skills.
    """
    title_l  = title.lower()
    # Filter out the title itself and obvious noise
    lines = [
        s for s in strings
        if s.lower() != title_l
        and len(s) > 1
        and not _NOISE_TEXT.search(s)
        and not (re.match(r'^https?://', s) and len(s) > 60)
    ]

    company  = 'Unknown'
    location = 'Not specified'
    salary   = 'Not specified'
    skills: list[str] = []

    for line in lines:
        # Salary check first (high confidence)
        if _is_salary(line):
            if salary == 'Not specified':
                salary = line.strip()[:100]
            continue

        # Location check (strict to avoid false positives)
        if _is_location(line):
            if location == 'Not specified':
                location = _LOCATION_LOOSE.search(line).group(0).strip()
            continue

        # Skills: line has bullet/comma separators
        if _SKILL_DELIM.search(line):
            for part in _SKILL_DELIM.split(line):
                part = part.strip()
                if (part.lower() not in _EMPLOYMENT_TYPES and
                    2 <= len(part) <= 50 and
                    not _is_salary(part)):
                    skills.append(part)
            continue

        # Company: first unmatched line
        # Remove star ratings from end (e.g., "Terrier Security 4.5★" → "Terrier Security")
        if company == 'Unknown' and 2 < len(line) < 100:
            co = _STAR_RATING.sub('', line).strip()
            # Remove trailing location suffix (e.g., "Company | India")
            co = re.sub(r'\s*[|•]\s*(India|Remote|Hybrid).*$', '', co, flags=re.I).strip()
            if co and len(co) > 2:
                company = co

    return {
        'company':  company[:100],
        'location': location[:80],
        'salary':   salary[:100],
        'skills':   list(dict.fromkeys(skills))[:15],  # deduplicated
    }


# ── Intent-driven job card extractor ──────────────────────────────────────────

def _extract_job_cards_from_soup(soup) -> list[dict]:
    """
    Intent-driven extraction: anchors are job titles, context is limited window.

    Method:
    1. Find <a href> tags whose text matches job title patterns
    2. Validate: exclude noise links, too long/short titles, duplicates
    3. Limited context: get parent <td>.strings ONLY (not entire ancestor)
    4. Parse context window into structured fields
    """
    job_cards: list[dict] = []
    seen_titles:  set[str] = set()

    for a_tag in soup.find_all('a', href=True):
        href  = a_tag['href'].strip()
        title = _clean(a_tag.get_text())

        # Validate job title
        if not (4 <= len(title) <= 120):
            continue
        if not _JOB_TITLE_RE.search(title):
            continue
        if _is_noise_link(href):
            continue
        if not href.startswith(('http://', 'https://')):
            continue

        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # Find nearest parent <td> — this is our limited context window
        parent_td = a_tag.find_parent('td')

        if parent_td:
            # Get strings from THIS td only (not ancestors)
            raw_strings = [_clean(str(s)) for s in parent_td.strings]
        else:
            # Fallback: get strings from direct parent element
            raw_strings = [_clean(str(s)) for s in a_tag.parent.strings]

        # Filter empty and noise strings
        context = [s for s in raw_strings if s and len(s) > 1 and not _NOISE_TEXT.search(s)]

        # Parse context window
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

    logger.info(f"Intent extraction found {len(job_cards)} job cards")
    return job_cards


def _build_display_text(soup, job_cards: list[dict]) -> str:
    """Build clean human-readable text for email display in UI."""
    lines: list[str] = []

    # Get a brief intro (first 2 meaningful lines before job listings)
    intro_count = 0
    for tag in (soup.find('body') or soup).children:
        if not hasattr(tag, 'get_text'):
            continue
        txt = _clean(tag.get_text(separator=' '))
        if not txt or len(txt) < 5 or _NOISE_TEXT.search(txt):
            continue
        # Stop when we reach job content
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
        # No job cards — fall back to clean text extraction
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
    Parse email body (HTML or plain text) into structured output.

    Pipeline:
    1. Detect if HTML (if not, return cleaned plain text)
    2. BeautifulSoup parse
    3. Remove noise: script/style/noscript/link[preload]/img
    4. Collect all href links
    5. Intent-driven job card extraction (anchor-based + limited context)
    6. Build readable display text

    Returns:
        text      : Clean readable text for UI display
        links     : All href links found
        job_links : Job-platform specific links
        job_cards : Structured [{role, company, location, salary, skills, link}]
        is_html   : Whether input was HTML
    """
    if not raw:
        return {'text': '', 'links': [], 'job_links': [], 'job_cards': [], 'is_html': False}

    # Clean invisible unicode before detection
    raw = _INVISIBLE.sub(' ', raw)

    is_html = bool(re.search(
        r'<(?:html|body|div|span|table|td|p|a)\b', raw, re.IGNORECASE
    ))

    if not is_html:
        text  = re.sub(r'\s+', ' ', raw).strip()
        text  = re.sub(r'\n{3,}', '\n\n', text)
        links = re.findall(r'https?://[^\s<>"\']{10,300}', text)
        return {
            'text':      text[:6000],
            'links':     list(dict.fromkeys(links))[:20],
            'job_links': [],
            'job_cards': [],
            'is_html':   False,
        }

    # Parse with BeautifulSoup
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _stdlib_fallback(raw)

    soup = BeautifulSoup(raw, 'html.parser')

    # ── Phase 1: Remove pure noise (no content value) ────────────────────────
    # script, style, noscript: executable/styling noise
    for tag in soup(['script', 'style', 'noscript', 'meta']):
        tag.decompose()
    # <link rel="preload"> tags: asset hints, zero content value
    for lt in soup.find_all('link'):
        lt.decompose()
    # Images: replace with alt text if meaningful, else remove entirely
    for img in soup.find_all('img'):
        alt = (_clean(img.get('alt', '') or img.get('title', '') or '')).strip()
        if alt and 2 < len(alt) < 40 and not re.search(r'logo|icon|pixel|banner|track', alt, re.I):
            img.replace_with(alt)
        else:
            img.decompose()

    # ── Phase 2: Collect all href links ──────────────────────────────────────
    all_links: list[str] = []
    job_links: list[str] = []
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href.startswith(('http://', 'https://')):
            continue
        if len(href) > 500:
            continue
        if _is_noise_link(href):
            continue
        all_links.append(href)
        if _is_job_link(href):
            job_links.append(href)

    all_links = list(dict.fromkeys(all_links))[:40]
    job_links = list(dict.fromkeys(job_links))[:15]

    # ── Phase 3: Intent-driven job card extraction ────────────────────────────
    job_cards = _extract_job_cards_from_soup(soup)

    # ── Phase 4: Build readable display text ─────────────────────────────────
    text = _build_display_text(soup, job_cards)

    return {
        'text':      text[:8000],
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
        return {'text': text[:8000], 'links': links, 'job_links': jlinks, 'job_cards': [], 'is_html': True}
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', raw)
        return {'text': text[:6000], 'links': [], 'job_links': [], 'job_cards': [], 'is_html': True}


# ── Convenience functions ──────────────────────────────────────────────────────

def clean_email_body(body: str, max_chars: int = 6000) -> str:
    """Get clean readable text from email body."""
    if not body:
        return ''
    return parse_email_html(body)['text'][:max_chars]


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