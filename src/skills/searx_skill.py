import httpx
from typing import Optional
from src.tools import registry

@registry.register
def search_searx(query: str) -> str:
    """
    Performs a search using the local SearXNG instance (searx.donahuenet.xyz) 
    and returns the top 3 results.
    """
    base_url = "https://searx.donahuenet.xyz"
    try:
        # Search parameters for SearXNG
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,duckduckgo"
        }
        
        # We use httpx for async-compatible synchronous requests in the tool registry
        response = httpx.get(f"{base_url}/search", params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])[:3] # Top 3
        if not results:
            return "No results found for your query on SearXNG."
            
        formatted = []
        for r in results:
            title = r.get("title", "No Title")
            url = r.get("url", "#")
            content = r.get("content", "No description available.")
            formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {content}")
            
        return "\n---\n".join(formatted)
        
    except httpx.HTTPStatusError as e:
        return f"SearXNG Error (HTTP {e.response.status_code}): {e}"
    except Exception as e:
        return f"Error connecting to SearXNG: {e}"
