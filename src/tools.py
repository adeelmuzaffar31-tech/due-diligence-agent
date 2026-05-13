"""Research tools the agent can call to gather company intelligence.

Each tool wraps a real external data source: Tavily for web search,
SEC EDGAR for filings, OFAC for sanctions screening.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from tavily import TavilyClient


# ============================================================
# Tool definitions (what Claude sees)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for news, press coverage, lawsuits, controversies, "
            "and general information about a company or person. Returns relevant "
            "article snippets with URLs. Use multiple specific queries from different "
            "angles for thorough research."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific search query. Examples: 'Acme Corp lawsuit 2024', "
                        "'Acme Corp CEO controversy', 'Acme Corp data breach'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_sec_edgar",
        "description": (
            "Search SEC EDGAR for public filings (10-K, 10-Q, 8-K, S-1). "
            "Only works for US public companies. Returns filing list with dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name to search EDGAR for",
                }
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "check_sanctions",
        "description": (
            "Check US Treasury OFAC sanctions list for an entity or person. "
            "Returns any matches with sanctions details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Entity or person name to check",
                }
            },
            "required": ["entity_name"],
        },
    },
]


# ============================================================
# Tool implementations
# ============================================================

def _tavily_client() -> TavilyClient:
    """Lazy-initialize the Tavily client so import doesn't fail without keys."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Get a free key at https://tavily.com/"
        )
    return TavilyClient(api_key=api_key)


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web via Tavily. Returns clean text, not raw HTML."""
    max_results = min(max(max_results, 1), 10)
    try:
        client = _tavily_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:1500],  # cap to control tokens
                }
                for r in response.get("results", [])
            ],
        }
    except Exception as exc:
        return {"query": query, "error": str(exc), "results": []}


def search_sec_edgar(company_name: str) -> dict[str, Any]:
    """Search SEC EDGAR full-text for company filings. Free public API."""
    try:
        response = httpx.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{company_name}"', "forms": "10-K,10-Q,8-K,S-1"},
            headers={
                "User-Agent": "Due Diligence Agent research@example.com",
                "Accept": "application/json",
            },
            timeout=15.0,
        )
        if response.status_code != 200:
            return {
                "company_name": company_name,
                "filings": [],
                "note": f"EDGAR returned status {response.status_code}",
            }
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])[:10]
        return {
            "company_name": company_name,
            "filings": [
                {
                    "form": h.get("_source", {}).get("form", ""),
                    "filing_date": h.get("_source", {}).get("file_date", ""),
                    "company": h.get("_source", {}).get("display_names", []),
                    "accession": h.get("_id", ""),
                }
                for h in hits
            ],
        }
    except Exception as exc:
        return {"company_name": company_name, "error": str(exc), "filings": []}


def check_sanctions(entity_name: str) -> dict[str, Any]:
    """Check US Treasury OFAC sanctions list. Free public API."""
    try:
        response = httpx.get(
            "https://sanctionssearch.ofac.treas.gov/api/PublicationPreview/exports/SDN_ENHANCED.JSON",
            timeout=20.0,
        )
        if response.status_code != 200:
            return {
                "entity_name": entity_name,
                "matches": [],
                "note": "OFAC service unreachable; treat as not-yet-screened",
            }
        # OFAC publishes the full list as JSON; we do a name match locally
        data = response.json()
        needle = entity_name.lower().strip()
        matches = []
        for record in data.get("publshInformation", {}).get("sdnEntries", [])[:5000]:
            primary_name = (record.get("lastName") or "").lower()
            if needle in primary_name or primary_name in needle:
                matches.append(
                    {
                        "name": record.get("lastName"),
                        "program": record.get("programList", {}).get("program", []),
                        "type": record.get("sdnType"),
                    }
                )
                if len(matches) >= 5:
                    break
        return {"entity_name": entity_name, "matches": matches}
    except Exception as exc:
        return {"entity_name": entity_name, "error": str(exc), "matches": []}


# ============================================================
# Dispatcher
# ============================================================

def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route tool calls from the agent to the right Python function."""
    if name == "web_search":
        return web_search(
            arguments["query"],
            max_results=arguments.get("max_results", 5),
        )
    if name == "search_sec_edgar":
        return search_sec_edgar(arguments["company_name"])
    if name == "check_sanctions":
        return check_sanctions(arguments["entity_name"])
    return {"error": f"Unknown tool: {name}"}
