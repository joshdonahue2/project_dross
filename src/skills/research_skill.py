"""Research related tools."""
from typing import Dict, Any, Optional

def get_wikipedia_summary(topic: str) -> str:
    """Returns a summary of a topic from Wikipedia."""
    import requests
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "No summary found.")
        return f"Wikipedia error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"
