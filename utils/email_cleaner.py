"""
Production-grade email body cleaner.

Fixes:
1. Extracts ALL anchor tag hrefs before stripping HTML
2. Removes CSS/JS noise completely
3. Returns readable plain text + extracted links
4. Handles nested skip tags correctly
5. Removes tracking pixels and invisible content
"""
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

# Tags whose content must be completely removed (including inner text)
_SKIP_TAGS = {
    "script", "style", "head", "meta", "link", "noscript",
    "iframe", "svg", "path", "defs", "symbol", "use",
}

# Tags that produce a newline in output
_BLOCK_TAGS = {
    "p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "blockquote", "article", "section",
    "header", "footer", "main", "aside",
}

# Domains that are job-related (for link filtering)
_JOB_LINK_DOMAINS = {
    "glassdoor.com", "glassdoor.co.in",
    "naukri.com", "linkedin.com", "indeed.com",
    "monster.com", "shine.com", "foundit.in",
    "instahyre.com", "unstop.com", "internshala.com",
    "wellfound.com", "angellist.com", "cutshort.io",
    "hirist.tech", "apna.co", "timesjobs.com",
    "careerbuilder.com", "ziprecruiter.com",
    "jobs.google.com", "greenhouse.io", "lever.co",
    "ashbyhq.com", "workable.com", "bamboohr.com",
}

# Patterns to remove from cleaned text
_NOISE_PATTERNS = [
    r'If you (?:are unable|cannot) (?:view|see) this email[^\n]*',
    r'View (?:this email|in browser)[^\n]*',
    r'To unsubscribe[^\n]*',
    r'You (?:are|were) (?:receiving|subscribed)[^\n]*',
    r'©\s*\d{4}[^\n]*',
    r'\[image(?::[^\]]+)?\]',
    r'Click here to[^\n]{0,80}',
    r'Privacy Policy[^\n]*',
    r'Terms of (?:Service|Use)[^\n]*',
    r'\d+px[^\n]{0,40}',          # CSS remnants
    r'display:\s*none[^\n]*',      # CSS remnants
    r'color:\s*#[0-9a-fA-F]+',    # CSS color values
    r'font-(?:size|family|weight)[^\n]*',  # CSS font properties
    r'background-color[^\n]*',
    r'https?://[^\s]{120,}',       # extremely long URLs (tracking)
    r'[a-f0-9]{40,}',              # SHA hashes (tracking IDs)
    r'==[a-zA-Z0-9+/]{20,}==',    # base64 blobs
]


class _EmailParser(HTMLParser):
    """
    Two-pass email HTML parser.
    Pass 1: collect all href links from anchor tags
    Pass 2: extract clean readable text
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._text_parts:  list[str] = []
        self._links:       list[str] = []
        self._skip_depth:  int       = 0
        self._in_anchor:   bool      = False
        self._current_href: str      = ""

    def handle_starttag(self, tag: str, attrs):
        tag_l = tag.lower()

        # Track skip-tag nesting depth
        if tag_l in _SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        # Extract href from anchor tags
        if tag_l == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href and href.startswith(("http://", "https://")):
                self._links.append(href.strip())
                self._current_href = href.strip()
            self._in_anchor = True

        # Block-level tags add newline for readability
        if tag_l in _BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str):
        tag_l = tag.lower()
        if tag_l in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag_l == "a":
            self._in_anchor = False
            self._current_href = ""

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text and len(text) > 1:
            # Skip invisible CSS class names and single chars
            if not re.match(r'^[{};:@]', text):
                self._text_parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._text_parts)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()

    def get_links(self) -> list[str]:
        # Deduplicate while preserving order
        seen, unique = set(), []
        for link in self._links:
            if link not in seen:
                seen.add(link)
                unique.append(link)
        return unique


def _remove_noise(text: str) -> str:
    """Remove common email boilerplate after HTML stripping."""
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_email_html(raw: str) -> dict:
    """
    Full email HTML parsing.

    Returns:
    {
      "text":       clean readable text,
      "links":      all hrefs extracted from anchor tags,
      "job_links":  filtered job-related links only,
      "is_html":    bool
    }
    """
    if not raw:
        return {"text": "", "links": [], "job_links": [], "is_html": False}

    is_html_content = bool(
        re.search(r"<(?:html|body|div|span|table|p|a)\b", raw, re.IGNORECASE)
    )

    if not is_html_content:
        # Plain text — just clean whitespace
        text = re.sub(r"[ \t]{2,}", " ", raw).strip()
        return {
            "text":      text[:3000],
            "links":     _extract_plain_text_links(text),
            "job_links": [],
            "is_html":   False,
        }

    parser = _EmailParser()
    try:
        parser.feed(raw)
        text  = parser.get_text()
        links = parser.get_links()
    except Exception:
        # Fallback regex strip
        text  = re.sub(r"<[^>]+>", " ", raw)
        text  = re.sub(r"\s+", " ", text).strip()
        links = _URL_RE.findall(raw)

    text      = _remove_noise(text)
    job_links = _filter_job_links(links)

    return {
        "text":      text[:4000],
        "links":     links[:30],
        "job_links": job_links[:10],
        "is_html":   True,
    }


_URL_RE = re.compile(r'https?://[^\s<>"\'(){}\[\]\\,]{10,}')


def _extract_plain_text_links(text: str) -> list[str]:
    return list(dict.fromkeys(_URL_RE.findall(text)))[:20]


def _filter_job_links(links: list[str]) -> list[str]:
    """Keep only links pointing to job platforms."""
    job_links = []
    for link in links:
        try:
            domain = urlparse(link).netloc.lower().lstrip("www.")
            if any(jd in domain for jd in _JOB_LINK_DOMAINS):
                job_links.append(link)
        except Exception:
            pass
    # Also keep links with job-related URL patterns
    for link in links:
        if link not in job_links:
            if re.search(r'/job[s]?/|/career[s]?/|/apply|/opening|/position|/role', link, re.IGNORECASE):
                job_links.append(link)
    return list(dict.fromkeys(job_links))[:10]


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """
    Convenience function — returns cleaned text only.
    Use parse_email_html() when you also need links.
    """
    if not body:
        return ""
    result = parse_email_html(body)
    return result["text"][:max_chars]


def extract_links_from_email(body: str) -> dict:
    """
    Extract all links and job-specific links from email body.
    Returns {"all_links": [...], "job_links": [...]}
    """
    result = parse_email_html(body)
    return {
        "all_links": result["links"],
        "job_links": result["job_links"],
    }


def is_html(text: str) -> bool:
    return bool(re.search(r"<(?:html|body|div|span|p|a)\b", text, re.IGNORECASE))
