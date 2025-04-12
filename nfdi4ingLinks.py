import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time

BASE_URL = "https://nfdi4ing.de"
visited = set()
all_links = set()

EXCLUDE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')

def normalize_url(href):
    parsed = urlparse(href)
    # Remove fragments and query parameters
    cleaned = parsed._replace(fragment="", query="")
    return urlunparse(cleaned)

def is_internal_link(link):
    return link.startswith(BASE_URL) and not link.lower().endswith(EXCLUDE_EXTENSIONS)

def extract_links(url):
    try:
        response = requests.get(url, timeout=10)
        if not response.ok:
            print(f"Failed to retrieve: {url}")
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = urljoin(url, a_tag['href'])  # resolves relative URLs
            href = normalize_url(href)
            if is_internal_link(href):
                links.add(href)
        return links
    except Exception as e:
        print(f"Error on {url}: {e}")
        return []

def crawl(url):
    url = normalize_url(url)
    if url in visited:
        return
    visited.add(url)
    print(f"Crawling: {url}")
    links = extract_links(url)
    all_links.update(links)
    for link in links:
        if link not in visited:
            # time.sleep(0.5)  # avoid hammering the server
            crawl(link)

# Start crawling
crawl(BASE_URL)

# Save results
print("\nAll collected links:")
for link in sorted(all_links):
    print(link)



with open("nfdi4ing_links.txt", "w") as f:
    for link in sorted(all_links):
        f.write(link + "\n")
