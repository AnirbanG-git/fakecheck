import requests
from typing import List, Dict

WIKI_API = "https://en.wikipedia.org/w/api.php"

def search_wikipedia(queries: List[str], top_k: int = 5) -> List[Dict]:
    results = []
    seen_titles = set()
    headers = {
        # 👇 Wikipedia requires a descriptive User-Agent with contact info or project URL
        "User-Agent": "FakeCheckBot/1.0 (https://github.com/your-repo; contact: youremail@example.com)"
    }

    for q in queries:
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": q,
            "srlimit": top_k
        }
        # add headers + error handling
        r = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
        if r.status_code == 403:
            print(f"[WARN] Wikipedia rejected the request (403). Possibly missing/invalid User-Agent for query: {q}")
            continue
        r.raise_for_status()

        data = r.json()
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            snippet = (
                hit.get("snippet", "")
                .replace('<span class="searchmatch">', "")
                .replace("</span>", "")
            )
            pageid = hit.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}"
            results.append({
                "source": "wikipedia",
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": float(hit.get("score", 0.0))
            })

    return results[:top_k]
