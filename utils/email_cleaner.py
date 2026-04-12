"""
Production Email Cleaner — Structure-Aware DOM Parsing.

Architecture:
  HTML → DOM traversal → job cards (structured) + clean text
  NOT: HTML → soup.get_text() → noisy text

Key design:
- Use soup.strings to get text nodes in order (respects <br> separators)
- Extract job cards by finding <a> tags with job-title text
- Parse context from parent <td>.strings — gives company/location/salary/skills in order
- Build readable display text from structured job cards
"""
import re
from urllib.parse import urlparse
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

_JOB_DOMAINS = {
    "glassdoor.com","glassdoor.co.in","naukri.com","linkedin.com",
    "indeed.com","monster.com","shine.com","foundit.in","instahyre.com",
    "unstop.com","internshala.com","wellfound.com","cutshort.io",
    "hirist.tech","apna.co","timesjobs.com","careerbuilder.com",
    "ziprecruiter.com","greenhouse.io","lever.co","ashbyhq.com",
    "workable.com","jobs.google.com","simplyhired.com",
}

_SKIP_LINK_PATTERNS = re.compile(
    r'manage.settings|unsubscribe|privacy|terms.of|help.center|'
    r'contact.us|about.us|brand.views|tracking|pixel|logomark',
    re.IGNORECASE
)

_JOB_TITLE_RE = re.compile(
    r'\b(engineer|developer|analyst|scientist|manager|intern|designer|'
    r'architect|lead|senior|junior|associate|consultant|specialist|'
    r'director|coordinator|fullstack|full.stack|frontend|backend|devops|'
    r'sde|swe|data|ml|ai|software|machine.learning|generative|applied|'
    r'research|product|cloud|security|nlp|computer.vision)\b',
    re.IGNORECASE
)

_STAR_RE     = re.compile(r'\s*\d+\.\d+\s*[★☆\*\u2605\u2606]?\s*$')
_SALARY_RE   = re.compile(
    r'[₹$€£]?\s*\d+[KkLl]?\s*[-–]\s*[₹$€£]?\s*\d+[KkLl]?'
    r'(?:\s*\([^)]{0,40}\))?',
    re.IGNORECASE
)
_CITIES_RE   = re.compile(
    r'\b(bangalore|bengaluru|mumbai|delhi|hyderabad|chennai|pune|kolkata|'
    r'noida|gurgaon|gurugram|ahmedabad|jaipur|kochi|indore|lucknow|bhopal|'
    r'nagpur|surat|india|remote|work\s+from\s+home|wfh|hybrid|onsite)\b',
    re.IGNORECASE
)
_SKILL_DELIM = re.compile(r'[•·,;|]')
_INVISIBLE   = re.compile(r'[\u200b\u200c\u200d\ufeff\u00ad\u00a0\u2028\u2029\xa0]')
_NOISE_TEXT  = re.compile(
    r'(unsubscribe|manage\s+settings|privacy\s+policy|terms\s+of|'
    r'this\s+message\s+was\s+sent|you\s+are\s+receiving|if\s+you\s+cannot|'
    r'view\s+in\s+browser|click\s+here\s+to|©\s*\d{4}|all\s+rights\s+reserved|'
    r'add\s+.*to\s+your\s+address|do\s+not\s+reply)',
    re.IGNORECASE
)

_EMPLOYMENT_TYPES = {
    'full-time','part-time','contract','freelance','permanent',
    'temporary','fixed-term','full time','part time','apprenticeship',
}


def _clean(text: str) -> str:
    """Remove invisible chars, normalize whitespace."""
    text = _INVISIBLE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _is_noise_link(href: str) -> bool:
    return bool(_SKIP_LINK_PATTERNS.search(href))


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


def _get_td_strings(td_tag) -> list[str]:
    """
    Get all text strings from a <td> in order.
    Uses .strings which respects <br> line breaks.
    Filters out invisible/empty strings.
    """
    lines = []
    for s in td_tag.strings:
        txt = _clean(str(s))
        if not txt or len(txt) <= 1:
            continue
        if _NOISE_TEXT.search(txt):
            continue
        if re.match(r'^https?://', txt) and len(txt) > 80:
            continue
        lines.append(txt)
    return lines


def _parse_job_context(title: str, context_lines: list[str]) -> dict:
    """
    Parse ordered context lines from job card <td> into structured fields.
    Order in Glassdoor emails: title → company (with rating) → location → salary → skills
    """
    title_lower = title.lower()
    lines = [l for l in context_lines if l.lower() != title_lower and len(l) > 1]

    company  = "Unknown"
    location = "Not specified"
    salary   = "Not specified"
    skills   = []

    for line in lines:
        # Salary: contains currency + number range
        if _SALARY_RE.search(line):
            salary = line.strip()[:100]
            continue

        # Location: short line with city/remote keyword
        if _CITIES_RE.search(line) and len(line) < 50:
            m = _CITIES_RE.search(line)
            location = m.group(0).strip()
            continue

        # Skills: line with bullet separators
        if _SKILL_DELIM.search(line):
            parts = [p.strip() for p in _SKILL_DELIM.split(line) if p.strip()]
            for p in parts:
                if (p.lower() not in _EMPLOYMENT_TYPES and
                    2 <= len(p) <= 50 and
                    not _SALARY_RE.search(p)):
                    skills.append(p)
            continue

        # Company: first unmatched line (remove star rating from end)
        if company == "Unknown" and 2 < len(line) < 100:
            co = _STAR_RE.sub('', line).strip()
            # Remove "| India" or "| Remote" suffixes
            co = re.sub(r'\s*\|\s*(India|Remote|Hybrid).*$', '', co, flags=re.I).strip()
            # Must not be just a city name
            if co and len(co) > 2 and not _CITIES_RE.fullmatch(co.lower()):
                company = co

    return {
        "role":     title,
        "company":  company,
        "location": location,
        "salary":   salary,
        "skills":   list(dict.fromkeys(skills))[:15],  # deduplicate
    }


def _extract_job_cards(soup) -> list[dict]:
    """
    Extract structured job cards from parsed HTML DOM.
    
    Method:
    1. Find every <a href> whose text looks like a job title
    2. Climb up to find parent <td>
    3. Use td.strings to get ordered text lines
    4. Parse lines into structured fields
    """
    job_cards = []
    seen      = set()

    for a_tag in soup.find_all("a", href=True):
        href  = a_tag["href"].strip()
        title = _clean(a_tag.get_text())

        # Validate job title
        if not (4 <= len(title) <= 120):
            continue
        if not _JOB_TITLE_RE.search(title):
            continue
        if _is_noise_link(href):
            continue
        if not href.startswith(("http://","https://")):
            continue

        title_key = title.lower()
        if title_key in seen:
            continue
        seen.add(title_key)

        # Find parent <td> (job card container)
        parent_td = None
        node      = a_tag.parent
        for _ in range(8):
            if node is None:
                break
            if node.name == "td":
                parent_td = node
                break
            node = node.parent

        if parent_td is None:
            # No td found — use direct parent
            context_lines = [title]
        else:
            context_lines = _get_td_strings(parent_td)

        if not context_lines:
            context_lines = [title]

        # Parse structured fields from context
        parsed = _parse_job_context(title, context_lines)
        parsed["link"]      = href
        parsed["all_links"] = [href]
        parsed["email_id"]  = ""
        parsed["email_subject"] = ""
        parsed["source"]    = "dom_structure"

        job_cards.append(parsed)

    logger.info(f"DOM extraction found {len(job_cards)} job cards")
    return job_cards


def _build_display_text(soup, job_cards: list[dict]) -> str:
    """Build clean readable text for UI display."""
    lines = []

    # Get email intro/subject text (first few meaningful lines)
    body = soup.find("body") or soup
    intro_added = 0
    for elem in body.children:
        if not hasattr(elem, 'get_text'):
            continue
        txt = _clean(elem.get_text(separator=' '))
        if not txt or len(txt) < 5:
            continue
        if _NOISE_TEXT.search(txt):
            continue
        # Stop at first job title
        if any(jc["role"].lower() in txt.lower() for jc in job_cards[:1]):
            break
        lines.append(txt)
        intro_added += 1
        if intro_added >= 2:
            break

    if lines:
        lines.append("")

    # Structured job listing
    if job_cards:
        lines.append(f"📋 {len(job_cards)} Job Listings:")
        lines.append("─" * 45)
        for i, card in enumerate(job_cards, 1):
            lines.append(f"\n{i}. {card['role']}")
            if card["company"] != "Unknown":
                lines.append(f"   🏢 {card['company']}")
            if card["location"] != "Not specified":
                lines.append(f"   📍 {card['location']}")
            if card["salary"] != "Not specified":
                lines.append(f"   💰 {card['salary']}")
            if card["skills"]:
                lines.append(f"   🔧 {', '.join(card['skills'][:8])}")
            if card["link"]:
                lines.append(f"   🔗 {card['link'][:90]}")
    else:
        # No job cards — fall back to readable text
        text = soup.get_text(separator="\n", strip=True)
        clean_lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) <= 1: continue
            if _NOISE_TEXT.search(line): continue
            if re.match(r'^https?://', line) and len(line) > 80: continue
            line = _clean(line)
            if line:
                clean_lines.append(line)
        lines.extend(clean_lines)

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_email_html(raw: str) -> dict:
    """
    Structure-aware email HTML parser.
    
    Returns:
      text       : Human-readable text for UI display
      links      : All href links
      job_links  : Job-platform links  
      job_cards  : Structured [{role, company, location, salary, skills, link}]
      is_html    : Whether input was HTML
    """
    if not raw:
        return {"text":"","links":[],"job_links":[],"job_cards":[],"is_html":False}

    is_html = bool(re.search(
        r"<(?:html|body|div|span|table|td|p|a)\b", raw, re.IGNORECASE
    ))

    if not is_html:
        text  = _clean(raw)
        text  = re.sub(r'\n{3,}', '\n\n', text)
        links = re.findall(r'https?://[^\s<>"\']{10,300}', text)
        return {
            "text":      text[:6000],
            "links":     list(dict.fromkeys(links))[:20],
            "job_links": [],
            "job_cards": [],
            "is_html":   False,
        }

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _stdlib_parse(raw)

    soup = BeautifulSoup(raw, "html.parser")

    # ── Phase 1: Remove pure noise (no content value) ─────────────────────────
    for tag in soup(["script","style","noscript","meta"]):
        tag.decompose()
    # Remove preload <link> tags (just asset hints, not content)
    for lt in soup.find_all("link"):
        lt.decompose()
    # Replace images with alt text (or remove if no alt)
    for img in soup.find_all("img"):
        alt = (img.get("alt","") or img.get("title","")).strip()
        if alt and 2 < len(alt) < 40:
            img.replace_with(f"[{alt}]")
        else:
            img.decompose()

    # ── Phase 2: Collect all links ────────────────────────────────────────────
    all_links, job_links = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith(("http://","https://")): continue
        if len(href) > 500: continue
        if _is_noise_link(href): continue
        all_links.append(href)
        if _is_job_link(href):
            job_links.append(href)

    all_links = list(dict.fromkeys(all_links))[:40]
    job_links = list(dict.fromkeys(job_links))[:15]

    # ── Phase 3: Structured DOM extraction ───────────────────────────────────
    job_cards = _extract_job_cards(soup)

    # Attach email info placeholder (filled by caller)
    for card in job_cards:
        if "email_id" not in card:
            card["email_id"]      = ""
            card["email_subject"] = ""

    # ── Phase 4: Build readable display text ─────────────────────────────────
    text = _build_display_text(soup, job_cards)

    return {
        "text":      text[:8000],
        "links":     all_links,
        "job_links": job_links,
        "job_cards": job_cards,
        "is_html":   True,
    }


def _stdlib_parse(raw: str) -> dict:
    """Fallback parser using stdlib HTMLParser."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts, self.links = [], []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            t = tag.lower()
            if t in ("script","style","noscript"): self._skip += 1; return
            if self._skip: return
            if t == "a":
                h = dict(attrs).get("href","")
                if h.startswith(("http://","https://")) and not _is_noise_link(h):
                    self.links.append(h)
            if t in ("br","p","div","tr","li","h1","h2","h3","h4","td"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag.lower() in ("script","style","noscript") and self._skip > 0:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                txt = _clean(data)
                if txt and not _NOISE_TEXT.search(txt):
                    self.parts.append(txt)

        def get_text(self):
            return re.sub(r'\n{3,}','\n\n',"\n".join(self.parts)).strip()

    p = _P()
    try:
        p.feed(raw)
        text   = p.get_text()
        links  = list(dict.fromkeys(p.links))[:40]
        jlinks = [l for l in links if _is_job_link(l)]
        return {"text":text[:8000],"links":links,"job_links":jlinks,"job_cards":[],"is_html":True}
    except Exception:
        text = re.sub(r'<[^>]+',' ',raw)
        return {"text":text[:6000],"links":[],"job_links":[],"job_cards":[],"is_html":True}


def clean_email_body(body: str, max_chars: int = 6000) -> str:
    """Get clean readable text from email body."""
    if not body:
        return ""
    result = parse_email_html(body)
    return result["text"][:max_chars]


def extract_job_cards(body: str) -> list[dict]:
    """Extract structured job cards from email HTML."""
    if not body:
        return []
    return parse_email_html(body).get("job_cards", [])


def extract_links_from_email(body: str) -> dict:
    result = parse_email_html(body)
    return {"all_links": result["links"], "job_links": result["job_links"]}


def is_html(text: str) -> bool:
    return bool(re.search(r"<(?:html|body|div|span|p|a|table)\b", text, re.IGNORECASE))