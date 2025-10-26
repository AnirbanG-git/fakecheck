import requests
from typing import List, Dict

CSE_API = "https://www.googleapis.com/customsearch/v1"

def search_cse(queries: List[str], api_key: str, cx: str, top_k: int = 5) -> List[Dict]:
    results = []
    seen_links = set()
    for q in queries:
        params = {"q": q, "key": api_key, "cx": cx, "num": min(top_k, 10)}
        r = requests.get(CSE_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", []):
            link = item.get("link")
            if not link or link in seen_links: 
                continue
            seen_links.add(link)
            results.append({
                "source": "google_cse",
                "title": item.get("title",""),
                "snippet": item.get("snippet",""),
                "url": link
            })
    return results[:top_k]
