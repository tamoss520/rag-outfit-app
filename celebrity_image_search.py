# celebrity_image_search.py
import os
import requests

def serpapi_search_images(query: str, count: int = 4):
    """
    Returns list of dicts with thumbnailUrl, contentUrl, hostPageUrl, name, source.
    """
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return [], "SERPAPI_KEY missing"

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": "0",
        "api_key": key
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], f"HTTP error: {e}"

    results = []
    for it in data.get("images_results", [])[:count]:
        results.append({
            "thumbnailUrl": it.get("thumbnail"),
            "contentUrl": it.get("original"),
            "hostPageUrl": it.get("link"),
            "name": it.get("title"),
            "source": it.get("source")
        })
    if not results:
        return [], "No images_results in response"
    return results, None
