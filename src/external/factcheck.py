# src/external/factcheck.py
from typing import List, Dict, Any, Optional
import time
import requests

FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def _call_factcheck(
    query: str,
    api_key: str,
    language_code: str = "en",
    page_token: Optional[str] = None,
    page_size: int = 10,
    timeout_s: int = 12,
) -> Dict[str, Any]:
    params = {
        "key": api_key,
        "query": query,
        "languageCode": language_code,
        "pageSize": page_size,
    }
    if page_token:
        params["pageToken"] = page_token
    r = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def search_fact_checks(
    queries: List[str],
    api_key: str,
    top_k: int = 10,
    language_code: str = "en",
    max_pages: int = 3,
    retries: int = 3,
    backoff_base: float = 0.8,
    timeout_s: int = 12,
    min_results_to_stop: int = 6,
) -> List[Dict[str, Any]]:
    """
    Try short, entity-centric queries first. Retry transient failures and paginate.
    """
    if not api_key:
        return []

    results: List[Dict[str, Any]] = []
    seen_urls = set()

    for q in queries:
        page_token = None
        for page in range(max_pages):
            # retry loop
            for attempt in range(retries):
                try:
                    data = _call_factcheck(
                        query=q,
                        api_key=api_key,
                        language_code=language_code,
                        page_token=page_token,
                        page_size=min(10, top_k),
                        timeout_s=timeout_s,
                    )
                    break
                except requests.RequestException as e:
                    # exponential backoff with jitter
                    sleep_s = (backoff_base ** attempt) * (1.0 + 0.5 * attempt)
                    time.sleep(min(4.0, sleep_s))
            else:
                # all retries failed; try next query
                break

            claims = data.get("claims", [])
            for c in claims:
                for cr in c.get("claimReview", []) or []:
                    url = cr.get("url") or ""
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append({
                        "type": "ClaimReview",
                        "text": c.get("text"),
                        "claimant": c.get("claimant"),
                        "claimDate": c.get("claimDate"),
                        "publisher": (cr.get("publisher") or {}).get("name"),
                        "title": cr.get("title"),
                        "url": url,
                        "reviewDate": cr.get("reviewDate"),
                        "textualRating": cr.get("textualRating"),
                    })
                    if len(results) >= top_k:
                        return results

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        # early stop if we already have enough reviews
        if len(results) >= min_results_to_stop:
            break

    return results[:top_k]
