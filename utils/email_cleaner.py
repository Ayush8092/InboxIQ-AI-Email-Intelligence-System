"""
Production-grade email body cleaner.

Uses BeautifulSoup for proper HTML parsing (handles real Gmail HTML correctly).
Falls back to stdlib HTMLParser if beautifulsoup4 not available.

Fixes:
1. Full HTML body extraction (not just snippets)
2. Anchor tag href extraction before stripping
3. Job-domain link detection
4. CSS/JS completely removed
5. Broken word repair from bad PDF extraction
"""
import re
from urllib.parse import urlparse
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

# Job platform domains — used to identify apply links
_JOB_DOMAINS = {
    "glassdoor.com", "glassdoor.co.in", "naukri.com",
    "linkedin.com", "indeed.com", "monster.com",
    "shine.com", "foundit.in", "instahyre.com",
    "unstop.com", "internshala.com", "wellfound.com",
    "cutshort.io", "hirist.tech", "apna.co",
    "timesjobs.com", "careerbuilder.com", "ziprecruiter.com",
    "greenhouse.io", "lever.co", "ashbyhq.com",
    "workable.com", "jobs.google.com", "simplyhired.com",
}

_NOISE_PATTERNS = [
    r'If you (?:are unable|cannot) (?:view|see) this email[^\n]*',
    r'View (?:this email|in browser)[^\n]*',
    r'To unsubscribe[^\n]*',
    r'You (?:are|were) (?:receiving|subscribed)[^\n]*',
    r'©\s*\d{4}[^\n]*',
    r'\[image(?::[^\]]+)?\]',
    r'https?://[^\s]{150,}',
    r'[a-f0-9]{40,}',
]


def _parse_with_beautifulsoup(html: str) -> tuple[str, list[str], list[str]]:
    """Parse HTML using BeautifulSoup — best quality."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags completely
    for tag in soup(["script", "style", "head", "meta",
                     "link", "noscript", "iframe", "img"]):
        tag.decompose()

    # Extract all links BEFORE getting text
    all_links  = []
    job_links  = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("http://", "https://")):
            all_links.append(href)
            try:
                domain = urlparse(href).netloc.lower().lstrip("www.")
                if any(jd in domain for jd in _JOB_DOMAINS):
                    job_links.append(href)
                elif re.search(
                    r'/job[s]?/|/career[s]?/|/apply|/opening|/position|/role|jobid',
                    href, re.IGNORECASE
                ):
                    job_links.append(href)
            except Exception:
                pass

    # Get clean text
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # Deduplicate links
    all_links = list(dict.fromkeys(all_links))[:30]
    job_links = list(dict.fromkeys(job_links))[:10]

    return text, all_links, job_links


def _parse_with_stdlib(html: str) -> tuple[str, list[str], list[str]]:
    """Fallback parser using stdlib HTMLParser."""
    from html.parser import HTMLParser

    _SKIP = {"script","style","head","meta","link","noscript","iframe","svg"}
    _BLOCK = {"p","br","div","li","h1","h2","h3","h4","h5","h6","tr","td","th"}

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._parts      = []
            self._links      = []
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            t = tag.lower()
            if t in _SKIP:
                self._skip_depth += 1
                return
            if self._skip_depth:
                return
            if t == "a":
                d = dict(attrs)
                h = d.get("href","")
                if h.startswith(("http://","https://")):
                    self._links.append(h)
            if t in _BLOCK:
                self._parts.append("\n")

        def handle_endtag(self, tag):
            if tag.lower() in _SKIP and self._skip_depth:
                self._skip_depth -= 1

        def handle_data(self, data):
            if not self._skip_depth:
                t = data.strip()
                if t:
                    self._parts.append(t)

        def get_text(self):
            raw = " ".join(self._parts)
            raw = re.sub(r"\n{3,}", "\n\n", raw)
            raw = re.sub(r"[ \t]{2,}", " ", raw)
            return raw.strip()

    p = _Parser()
    try:
        p.feed(html)
        text  = p.get_text()
        links = list(dict.fromkeys(p._links))[:30]
        job_links = []
        for lnk in links:
            try:
                domain = urlparse(lnk).netloc.lower().lstrip("www.")
                if any(jd in domain for jd in _JOB_DOMAINS):
                    job_links.append(lnk)
            except Exception:
                pass
        return text, links, job_links[:10]
    except Exception:
        plain = re.sub(r"<[^>]+>", " ", html)
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain, [], []


def _remove_noise(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _repair_broken_words(text: str) -> str:
    """
    Fix PDF extraction issues like 'learn continu ously' → 'learn continuously'.
    Joins words split by space when the split doesn't form real words.
    """
    # Fix common PDF word-break patterns
    text = re.sub(r'(\w{3,})\s+ously\b', r'\1ously', text)
    text = re.sub(r'(\w{3,})\s+tion\b',  r'\1tion',  text)
    text = re.sub(r'(\w{3,})\s+ing\b',   r'\1ing',   text)
    text = re.sub(r'(\w{3,})\s+ment\b',  r'\1ment',  text)
    text = re.sub(r'(\w{3,})\s+ness\b',  r'\1ness',  text)
    text = re.sub(r'(\w{3,})\s+ity\b',   r'\1ity',   text)
    text = re.sub(r'(\w{3,})\s+ance\b',  r'\1ance',  text)
    text = re.sub(r'(\w{3,})\s+ence\b',  r'\1ence',  text)
    return text


def parse_email_html(raw: str) -> dict:
    """
    Full email HTML parsing.
    Returns {text, links, job_links, is_html}
    """
    if not raw:
        return {"text": "", "links": [], "job_links": [], "is_html": False}

    is_html = bool(re.search(
        r"<(?:html|body|div|span|table|p|a|td)\b", raw, re.IGNORECASE
    ))

    if not is_html:
        text = re.sub(r"[ \t]{2,}", " ", raw).strip()
        links = re.findall(r'https?://[^\s<>"\'(){}\[\]\\,]{10,}', text)
        return {
            "text":      text[:4000],
            "links":     links[:20],
            "job_links": [],
            "is_html":   False,
        }

    # Try BeautifulSoup first (best quality)
    try:
        text, all_links, job_links = _parse_with_beautifulsoup(raw)
    except ImportError:
        text, all_links, job_links = _parse_with_stdlib(raw)

    text = _remove_noise(text)

    return {
        "text":      text[:5000],
        "links":     all_links,
        "job_links": job_links,
        "is_html":   True,
    }


def clean_email_body(body: str, max_chars: int = 4000) -> str:
    """Returns cleaned plain text from email body."""
    if not body:
        return ""
    result = parse_email_html(body)
    return result["text"][:max_chars]


def extract_links_from_email(body: str) -> dict:
    result = parse_email_html(body)
    return {
        "all_links": result["links"],
        "job_links": result["job_links"],
    }


def is_html(text: str) -> bool:
    return bool(re.search(r"<(?:html|body|div|span|p|a)\b", text, re.IGNORECASE))