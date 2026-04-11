"""
Final production email cleaner.

Root cause confirmed:
- Email body is stored as full HTML (<!DOCTYPE html>...)
- Old cleaner removed <head> and too many container tags
- Glassdoor HTML has content in <body> inside nested <table>/<td>
- Fix: only remove script/style/noscript, keep everything else for text extraction
- Also extract all href links BEFORE any stripping
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

_NOISE_PATTERNS = [
    r'To unsubscribe[^\n]{0,150}',
    r'©\s*\d{4}[^\n]{0,100}',
    r'You (?:are|were) receiving this[^\n]{0,150}',
    r'If you (?:are unable|cannot) (?:view|see) this email[^\n]{0,150}',
    r'Please (?:do not reply|add .+ to your)[^\n]{0,150}',
    r'This (?:is a marketing|message was sent)[^\n]{0,150}',
    r'Manage settings[^\n]{0,100}',
    r'Privacy Policy[^\n]{0,100}',
    r'[a-f0-9]{40,}',  # tracking hashes
]


def _parse_html(html: str) -> tuple[str, list[str], list[str]]:
    """
    Parse HTML email body.
    Returns (clean_text, all_links, job_links).
    
    Key fix: only remove script/style/noscript tags.
    Do NOT remove head/link/meta — Glassdoor links are in <link rel="preload">
    but more importantly, job content is in <body> tables.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Step 1: Collect ALL href links before any modification
    all_links: list[str] = []
    job_links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("http://","https://")) and len(href) < 500:
            all_links.append(href)
            try:
                domain = urlparse(href).netloc.lower().lstrip("www.")
                is_job = any(jd in domain for jd in _JOB_DOMAINS)
                is_apply = re.search(
                    r'/job[s]?[-/]|/career|/apply|/opening|/position|'
                    r'jobid=|job_id=|jid=|listingId=',
                    href, re.IGNORECASE
                )
                if is_job or is_apply:
                    job_links.append(href)
            except Exception:
                pass

    # Step 2: Remove ONLY noise tags (script, style, noscript)
    # Do NOT remove head/meta/link — they don't add text anyway
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove images (no text content)
    for tag in soup.find_all("img"):
        tag.decompose()

    # Step 3: Extract text with structure
    text = soup.get_text(separator="\n", strip=True)

    # Step 4: Clean up text
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Skip empty lines and noise
        if not line:
            continue
        # Skip pure URL lines (tracking pixels etc)
        if re.match(r'^https?://', line) and len(line) > 100:
            continue
        # Skip lines that are just numbers or single chars
        if re.match(r'^[\d\s]{1,3}$', line):
            continue
        lines.append(line)

    text = "\n".join(lines)

    # Step 5: Remove noise patterns
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # Deduplicate links
    all_links = list(dict.fromkeys(all_links))[:40]
    job_links = list(dict.fromkeys(job_links))[:15]

    return text, all_links, job_links


def _parse_html_stdlib(html: str) -> tuple[str, list[str], list[str]]:
    """Fallback stdlib parser if BeautifulSoup not available."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.links: list[str] = []
            self._skip = 0
            self._skip_tags = {"script", "style", "noscript"}

        def handle_starttag(self, tag, attrs):
            t = tag.lower()
            if t in self._skip_tags:
                self._skip += 1
                return
            if self._skip:
                return
            if t == "a":
                href = dict(attrs).get("href", "")
                if href.startswith(("http://", "https://")):
                    self.links.append(href)
            if t in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag.lower() in self._skip_tags and self._skip > 0:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                t = data.strip()
                if t:
                    self.parts.append(t)

        def get_text(self):
            raw = "\n".join(self.parts)
            raw = re.sub(r"\n{3,}", "\n\n", raw)
            return re.sub(r"[ \t]{2,}", " ", raw).strip()

    p = _P()
    try:
        p.feed(html)
        text  = p.get_text()
        links = list(dict.fromkeys(p.links))[:40]
        job_links = [
            l for l in links
            if any(jd in urlparse(l).netloc.lower() for jd in _JOB_DOMAINS)
        ]
        return text, links, job_links[:15]
    except Exception:
        plain = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", plain).strip(), [], []


def parse_email_html(raw: str) -> dict:
    """
    Parse email body HTML → clean text + links.
    
    Handles:
    - Standard HTML (<!DOCTYPE html>...)
    - Partial HTML
    - Plain text
    """
    if not raw:
        return {"text": "", "links": [], "job_links": [], "is_html": False}

    # Check if it's HTML
    is_html = bool(re.search(
        r"<(?:html|body|div|span|table|td|p|a)\b",
        raw, re.IGNORECASE
    ))

    if not is_html:
        # Plain text — extract URLs directly
        text  = re.sub(r"[ \t]{2,}", " ", raw).strip()
        links = re.findall(r'https?://[^\s<>"\'(){}\[\]\\,]{10,300}', text)
        return {
            "text":      text[:6000],
            "links":     list(dict.fromkeys(links))[:20],
            "job_links": [],
            "is_html":   False,
        }

    # Parse HTML
    try:
        text, all_links, job_links = _parse_html(raw)
    except ImportError:
        text, all_links, job_links = _parse_html_stdlib(raw)
    except Exception as e:
        logger.warning(f"HTML parse error: {type(e).__name__} — using stdlib fallback")
        text, all_links, job_links = _parse_html_stdlib(raw)

    return {
        "text":      text[:7000],
        "links":     all_links,
        "job_links": job_links,
        "is_html":   True,
    }


def clean_email_body(body: str, max_chars: int = 6000) -> str:
    """Get clean readable text from email body."""
    if not body:
        return ""
    return parse_email_html(body)["text"][:max_chars]


def extract_links_from_email(body: str) -> dict:
    result = parse_email_html(body)
    return {"all_links": result["links"], "job_links": result["job_links"]}


def is_html(text: str) -> bool:
    return bool(re.search(r"<(?:html|body|div|span|p|a)\b", text, re.IGNORECASE))