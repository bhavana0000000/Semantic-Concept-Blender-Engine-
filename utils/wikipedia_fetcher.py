"""
Wikipedia data fetcher with caching, disambiguation handling, and error recovery.
"""
import re
import time
import functools
import requests

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

_cache: dict = {}


def _search_wikipedia(query: str) -> str | None:
    """Return the best-matching Wikipedia page title for a query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except Exception:
        pass
    return None


def _get_summary(title: str) -> dict:
    """Fetch the REST-v1 summary for a given Wikipedia title."""
    url = WIKI_API + requests.utils.quote(title)
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "ConceptExplainer/1.0"})
        if r.status_code == 404:
            return {"error": f'No Wikipedia article found for "{title}".'}
        r.raise_for_status()
        data = r.json()
        # Handle disambiguation pages
        if data.get("type") == "disambiguation":
            return {"error": f'"{title}" is a disambiguation page. Please be more specific.'}
        return data
    except requests.exceptions.Timeout:
        return {"error": "Wikipedia request timed out."}
    except Exception as e:
        return {"error": str(e)}


def _get_full_content(title: str) -> str:
    """Fetch plain-text extract of the full article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "titles": title,
        "format": "json",
        "exsectionformat": "plain",
    }
    try:
        r = requests.get(WIKI_SEARCH, params=params, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except Exception:
        pass
    return ""


def fetch_concept_data(concept: str) -> dict:
    """
    Main entry point. Returns a rich dict with:
      - title, summary, full_text, categories, url
    or {"error": "<message>"} on failure.
    """
    key = concept.strip().lower()
    if key in _cache:
        return _cache[key]

    # 1. Try direct lookup
    summary_data = _get_summary(concept)

    # 2. Fall back to search if not found
    if summary_data.get("error"):
        found_title = _search_wikipedia(concept)
        if found_title:
            summary_data = _get_summary(found_title)

    if summary_data.get("error"):
        return summary_data

    title = summary_data.get("title", concept)
    full_text = _get_full_content(title)

    result = {
        "title": title,
        "summary": summary_data.get("extract", ""),
        "full_text": full_text,
        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "categories": [],  # populated below if available
        "description": summary_data.get("description", ""),
    }

    # Extract section headings as pseudo-categories
    headings = re.findall(r"^==\s*(.+?)\s*==$", full_text, re.MULTILINE)
    result["sections"] = headings[:10]

    _cache[key] = result
    return result
