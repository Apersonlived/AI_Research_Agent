"""
search_provider.py :: web search routing
---------------------------------------------------
Priority:
  1. Tavily: 1,000 credits/month free, purpose-built for LLM agents,
           LangChain-native, returns clean pre-extracted content
  2. Jina: Completely free, no API key needed, no rate limit published,
           use as fallback when Tavily credits are low

Usage:
    from search_provider import search_web

    results = search_web("AI safety techniques 2026", max_results=3)
    # Returns list of {"url", "title", "content", "score"} dicts
"""

import os
import json
import urllib.request
import urllib.parse
from typing import List


def search_with_tavily(query: str, max_results: int = 3, advanced: bool = False) -> List[dict]:
    """
    Use Tavily API. Costs 1 credit per basic search, more for advanced.
    Free tier: 1,000 credits/month.
    """
    from tavily import TavilyClient
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise EnvironmentError("TAVILY_API_KEY not set")

    client = TavilyClient(api_key=key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced" if advanced else "basic",
        include_raw_content=False,
    )
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", "Untitled"),
            "content": r.get("content", "")[:800],
            "score": r.get("score", 0),
        }
        for r in response.get("results", [])
    ]

# fallback if Tavily doesnot work
def search_with_jina(query: str, max_results: int = 3) -> List[dict]:
    """
    Use Jina Search API (s.jina.ai), no API key required.
    Returns full page content (truncated). Slightly slower than Tavily.
    Docs: https://jina.ai/reader/
    """
    encoded = urllib.parse.quote(query)
    url = f"https://s.jina.ai/{encoded}"

    headers = {
        "Accept": "application/json",
        "X-Respond-With": "no-content",  # get snippets rather than full pages (for faster retrieval)
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"Jina search failed: {e}")

    results = data.get("data", [])[:max_results]
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", "Untitled"),
            "content": (r.get("description") or r.get("content") or "")[:800],
            "score": 1.0,  # Jina doesn't return relevance scores
        }
        for r in results
        if r.get("url")
    ]


def search_web(
    query: str,
    max_results: int = 3,
    provider: str = "auto",
) -> List[dict]:
    """
    Search the web and return LLM-ready results.

    provider="auto":  tries Tavily first, falls back to Jina
    provider="tavily": Tavily only (requires TAVILY_API_KEY)
    provider="jina":   Jina only (completely free, no key needed)

    Returns: list of {"url", "title", "content", "score"}
    """
    if provider == "jina":
        return search_with_jina(query, max_results)

    if provider == "tavily":
        return search_with_tavily(query, max_results)

    # auto: try Tavily, otherwise fall back to Jina
    if os.getenv("TAVILY_API_KEY"):
        try:
            return search_with_tavily(query, max_results)
        except Exception as e:
            print(f" Tavily failed ({e}), falling back to Jina...")
    else:
        print(" No TAVILY_API_KEY — using Jina (free, no key needed)")

    return search_with_jina(query, max_results)