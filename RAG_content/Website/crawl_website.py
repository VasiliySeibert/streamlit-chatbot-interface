import os
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Base URL to start crawling
BASE_URL = 'https://nfdi4ing.de/'
# Local output folder
OUTPUT_FOLDER = 'nfdi4ing_html'
# Maximum crawling depth (modify as needed)
MAX_DEPTH = 2

# A set to track visited URLs
visited = set()

def get_local_path(url):
    """
    Create the local file path based on the URL.
    """
    parsed = urlparse(url)
    # Use the path from URL; if empty, use '/'.
    path = parsed.path if parsed.path else '/'
    
    # If the path ends with "/" then we name it "index.html"
    # or if there is no file extension, assume it is a directory.
    if path.endswith('/'):
        path = path + 'index.html'
    else:
        if not os.path.splitext(path)[1]:
            path = path + '/index.html'
    
    local_path = os.path.join(OUTPUT_FOLDER, parsed.netloc, path.lstrip('/'))
    return local_path

def save_page(url, content):
    """
    Save HTML content to a file that mirrors the URL structure.
    """
    local_path = get_local_path(url)
    # Make sure the directory exists.
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Write the content to the file.
    with open(local_path, "w", encoding="utf-8") as file:
        file.write(content)
    
    print(f"Saved {url} -> {local_path}")

def crawl(url, max_depth, current_depth=0):
    """
    Recursively crawl the website starting from the given URL.
    If the page has already been saved, it is loaded from disk
    to extract further links.
    """
    if current_depth > max_depth:
        return

    # Avoid re-visiting URLs during this session.
    if url in visited:
        return
    visited.add(url)

    local_path = get_local_path(url)
    
    # If the file already exists, load its content from disk
    if os.path.exists(local_path):
        print(f"Already crawled, processing links from: {url}")
        try:
            with open(local_path, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading {local_path}: {str(e)}")
            return
    else:
        print(f"Crawling: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Skipping {url} due to status code: {response.status_code}")
                return
            content = response.text
            save_page(url, content)
        except Exception as e:
            print(f"Failed to fetch {url}: {str(e)}")
            return

    # Use BeautifulSoup to parse and extract all anchor links.
    soup = BeautifulSoup(content, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Convert relative URLs to absolute URLs.
        next_url = urljoin(url, href)
        # Filter out URLs that do not belong to the base domain.
        if urlparse(next_url).netloc != urlparse(BASE_URL).netloc:
            continue

        # Remove any fragment portion (after '#' symbol)
        next_url = next_url.split('#')[0]
        
        # Recursively crawl the next URL.
        crawl(next_url, max_depth, current_depth + 1)
        time.sleep(.2)  # Pause briefly between requests

if __name__ == '__main__':
    crawl(BASE_URL, MAX_DEPTH)
