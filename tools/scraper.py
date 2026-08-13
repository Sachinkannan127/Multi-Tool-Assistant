"""
Web Scraper Tool — extracts full page content and all links from a given URL.

Uses httpx for async HTTP requests and BeautifulSoup for HTML parsing.
"""

import re
import logging
from typing import Optional
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Browser-like headers to avoid bot blocking
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

_MAX_CONTENT_CHARS = 8_000   # Cap scraped body text
_MAX_LINKS = 30               # Max links to return
_TIMEOUT = 15                 # Seconds


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip blank lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:
        return False


def _scrape_sync(url: str) -> dict:
    """
    Synchronous scraper using httpx + BeautifulSoup.
    Returns {'title', 'text', 'links', 'meta_description', 'error'}.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Install httpx and beautifulsoup4."}

    if not _is_valid_url(url):
        return {"error": f"Invalid URL: '{url}'. Must start with http:// or https://."}

    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {"error": f"Unsupported content type: {content_type}. Only HTML pages are supported."}
            html = resp.text
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {_TIMEOUT}s. The server may be slow or unreachable."}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code} error for URL: {url}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title = soup.title.get_text(strip=True) if soup.title else "No title"

        # Meta description
        meta_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        meta_desc = ""
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe", "svg"]):
            tag.decompose()

        # Extract body text
        body = soup.find("main") or soup.find("article") or soup.find("body") or soup
        raw_text = body.get_text(separator="\n")
        text = _clean_text(raw_text)
        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + f"\n\n[...content truncated at {_MAX_CONTENT_CHARS} characters]"

        # Extract links
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute = urljoin(base_url, href)
            if absolute not in seen and _is_valid_url(absolute):
                link_text = a.get_text(strip=True) or absolute
                links.append({"text": link_text[:100], "url": absolute})
                seen.add(absolute)
            if len(links) >= _MAX_LINKS:
                break

        return {
            "title": title,
            "meta_description": meta_desc,
            "text": text,
            "links": links,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Scraping parse error for {url}: {e}")
        return {"error": f"Failed to parse page content: {str(e)}"}


@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrape a webpage to extract its full text content and all hyperlinks.

    Use this tool when:
    - The user provides a specific URL and asks you to read, summarize, or analyze it.
    - The user wants "internet data to fetch and give the summarized output".
    - The user asks to 'extract links', 'scrape a page', 'get data from a website',
      or 'find all links on' a webpage.
    - You need to retrieve detailed content from a specific page that web_search
      only partially covered.

    Input: A full URL starting with http:// or https://
    Output: Page title, meta description, body text content, and a list of extracted hyperlinks.

    DO NOT use this tool for general queries — use web_search instead.
    Only call this when a specific URL is provided or strongly implied.
    """
    result = _scrape_sync(url)

    if result.get("error"):
        return f"❌ Scrape failed: {result['error']}"

    # Format output for the LLM
    lines = []
    lines.append(f"# Scraped Page: {result['title']}")
    lines.append(f"🔗 URL: {url}")

    if result.get("meta_description"):
        lines.append(f"📝 Description: {result['meta_description']}")

    lines.append("\n## Page Content\n")
    lines.append(result["text"] or "(No readable text found)")

    if result.get("links"):
        lines.append(f"\n## Extracted Links ({len(result['links'])} found)\n")
        for i, link in enumerate(result["links"], 1):
            lines.append(f"{i}. [{link['text']}]({link['url']})")

    return "\n".join(lines)
