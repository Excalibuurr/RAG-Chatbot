# Utility for web search to fetch current job market trends
import requests
from bs4 import BeautifulSoup

def fetch_job_trends(query: str) -> list:
    # Example: Scrape Google search results for top skills
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        results.append(g.get_text())
    # Fallback: Just return first few results
    return results[:5] if results else ["No trends found. Try a different query."]
