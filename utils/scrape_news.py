import os
import requests
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Ensure your .env file has: JINA_API_KEY=your_key_here
API_TOKEN = os.getenv("JINA_API_KEY")

all_cs_items = json.load(open("cs_news_urls.json", "r", encoding="utf-8"))
urls = [item['news_url'] for item in all_cs_items]
output_dir = "techxplore_news_pages"

# # Your list of TechXplore URLs (non-sequential)
# urls = [
#     "https://techxplore.com/computer-sciences-news/page1.html",
#     "https://techxplore.com/computer-sciences-news/page5.html",
#     "https://techxplore.com/computer-sciences-news/page12.html",
#     # ... add more
# ]

# Output folder
os.makedirs(output_dir, exist_ok=True)

# Loop through URLs and download via Jina API
for i, url in tqdm(enumerate(urls)):

    jina_url = f"https://r.jina.ai/{url}"
    output_path = os.path.join(output_dir, f"242.html")

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    print(f"Fetching {url} → {output_path}")

    try:
        response = requests.get(jina_url, headers=headers, timeout=100)
        response.raise_for_status()  # Raises HTTPError if not 200 OK

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)

            print(f"✅ Saved: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch {url}: {e}")

    time.sleep(10)  # polite delay to avoid rate-limiting
