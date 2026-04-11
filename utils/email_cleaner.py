"""
Production email cleaner.

Root cause of "Could not extract readable text":
Gmail API returns email body as base64url-encoded HTML in payload.parts[].body.data
The body stored in DB is the raw base64 string, NOT decoded HTML.
This cleaner handles both cases:
1. Already-decoded HTML  
2. Base64url-encoded content that needs decoding first
3. Plain text emails
"""
import re
import base64
from urllib.parse import urlparse
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

_JOB_DOMAINS = {
    "glassdoor.com","glassdoor.co.in","naukri.com","linkedin.com",
    "indeed.com","monster.com","shine.com","foundit.in","instahyre.com",
    "unstop.com","internshala.com","wellfound.com","cutshort.io",
    "hirist.tech","apna.co","timesjobs.com","careerbuilder.com",
    "ziprecruiter.com","greenhouse.io","lever.co","ashbyhq.com",
    "workable.com","jobs.google.com","simplyhired.com","angellist.com",
}

_NOISE_PATTERNS = [
    r'To unsubscribe[^\n]*',
    r'You (?:are|were) (?:receiving|subscribed)[^\n]*',
    r'©\s*\d{4}[^\n]*',
    r'\[image(?::[^\]]+)?\]',
    r'https?://[^\s]{200,}',
    r'[a-f0-9]{50,}',
    r'If you (?:are unable|cannot) (?:view|see) this email[^\n]*',
    r'View (?:this email|in browser|web version)[^\n]*',
]


def _try_base64_decode(text: str) -> str | None:
    """
    Try to decode base64url string.
    Gmail stores email body as base64url in payload.parts[].body.data
    """
    if not text:
        return None
    # Gmail uses URL-safe base64 — replace - with + and _ with /
    try:
        cleaned = text.replace('-', '+').replace('_', '/')
        # Add padding
        padding = 4 - len(cleaned) % 4
        if padding != 4:
            cleaned += '=' * padding
        decoded = base64.b64decode(cleaned).decode('utf-8', errors='replace')
        # Only return if it looks like real content
        if len(decoded) > 50 and ('<' in decoded or len(decoded.split()) > 5):
            return decoded
    except Exception:
        pass
    return None


def _is_base64_like(text: str) -> bool:
    """Check if text looks like base64 encoded data."""
    if len(text) < 50:
        return False
    # Base64 strings are mostly alphanumeric + / + = with no spaces
    sample = text[:200].replace('\n','').replace('\r','').replace(' ','')
    b64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=-_')
    ratio = sum(1 for c in sample if c in b64_chars) / max(len(sample), 1)
    return ratio > 0.90 and ' ' not in text[:100]


def _extract_text_with_bs4(html: str) -> tuple[str, list[str], list[str]]:
    """Extract clean text and links using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script","style","head","meta","link","noscript","img"]):
        tag.decompose()

    # Collect all links before stripping
    all_links, job_links = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("http://","https://")):
            all_links.append(href)
            try:
                domain = urlparse(href).netloc.lower().lstrip("www.")
                if any(jd in domain for jd in _JOB_DOMAINS):
                    job_links.append(href)
                elif re.search(r'/job[s]?/|/career[s]?/|/apply|/opening|/position', href, re.I):
                    job_links.append(href)
            except Exception:
                pass

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    return (
        text,
        list(dict.fromkeys(all_links))[:30],
        list(dict.fromkeys(job_links))[:10],
    )


def _extract_text_stdlib(html: str) -> tuple[str, list[str], list[str]]:
    """Fallback text extraction using stdlib."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts, self.links = [], []
            self._skip = 0
            self._skip_tags = {"script","style","head","noscript"}
            self._block = {"p","br","div","li","h1","h2","h3","h4","tr","td"}

        def handle_starttag(self, tag, attrs):
            t = tag.lower()
            if t in self._skip_tags: self._skip += 1; return
            if self._skip: return
            if t == "a":
                d = dict(attrs)
                h = d.get("href","")
                if h.startswith(("http://","https://")): self.links.append(h)
            if t in self._block: self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag.lower() in self._skip_tags and self._skip: self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                t = data.strip()
                if t: self.parts.append(t)

        def get_text(self):
            raw = " ".join(self.parts)
            raw = re.sub(r"\n{3,}","\n\n",raw)
            return re.sub(r"[ \t]{2,}"," ",raw).strip()

    p = _P()
    try:
        p.feed(html)
        text  = p.get_text()
        links = list(dict.fromkeys(p.links))[:30]
        job_links = []
        for lnk in links:
            try:
                domain = urlparse(lnk).netloc.lower().lstrip("www.")
                if any(jd in domain for jd in _JOB_DOMAINS): job_links.append(lnk)
            except: pass
        return text, links, job_links[:10]
    except Exception:
        plain = re.sub(r"<[^>]+>"," ",html)
        return re.sub(r"\s+"," ",plain).strip(), [], []


def _remove_noise(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def parse_email_html(raw: str) -> dict:
    """
    Parse email body — handles base64, HTML, and plain text.
    Returns {text, links, job_links, is_html}
    """
    if not raw:
        return {"text":"","links":[],"job_links":[],"is_html":False}

    # Step 1: Try to decode if it looks like base64
    decoded = None
    if _is_base64_like(raw):
        decoded = _try_base64_decode(raw)
        if decoded:
            raw = decoded

    # Step 2: Check if it's HTML
    is_html = bool(re.search(r"<(?:html|body|div|span|table|p|a|td)\b", raw, re.IGNORECASE))

    if not is_html:
        # Plain text
        text  = re.sub(r"[ \t]{2,}", " ", raw).strip()
        links = re.findall(r'https?://[^\s<>"\'(){}\[\]\\,]{10,}', text)
        return {
            "text":      text[:5000],
            "links":     links[:20],
            "job_links": [],
            "is_html":   False,
        }

    # Step 3: Parse HTML
    try:
        text, all_links, job_links = _extract_text_with_bs4(raw)
    except ImportError:
        text, all_links, job_links = _extract_text_stdlib(raw)

    text = _remove_noise(text)

    return {
        "text":      text[:6000],
        "links":     all_links,
        "job_links": job_links,
        "is_html":   True,
    }


def clean_email_body(body: str, max_chars: int = 5000) -> str:
    """Returns clean readable text from email body (handles base64 + HTML)."""
    if not body:
        return ""
    result = parse_email_html(body)
    return result["text"][:max_chars]


def extract_links_from_email(body: str) -> dict:
    result = parse_email_html(body)
    return {"all_links": result["links"], "job_links": result["job_links"]}


def is_html(text: str) -> bool:
    return bool(re.search(r"<(?:html|body|div|span|p|a)\b", text, re.IGNORECASE))