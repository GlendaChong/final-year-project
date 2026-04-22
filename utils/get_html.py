import pandas as pd
import sys, os
import json
from tqdm import tqdm
import re


sys.path.append(os.getcwd())

categories = {
    "Computer Sciences", "Software", "Robotics", 
    "Energy & Green Tech", "Consumer & Gadgets", 
    "Engineering", "Hardware", "Business", "Other"
}

def extract_cs_excerpts(file_path):
    processed_excerpts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() in categories:
            start = max(i - 2, 0)
            end = min(i + 6, len(lines))  # up to 8 lines
            excerpt = lines[start:end]

            # Process based on presence of "###"
            excerpt_text = ''.join(excerpt)
            if "###" in excerpt_text:
                selected = []
                if len(excerpt) > 0:
                    selected.append(excerpt[0])
                if len(excerpt) > 4:
                    selected.append(excerpt[4])
                processed_excerpts.append(''.join(selected))
            else:
                if len(excerpt) > 6:
                    processed_excerpts.append(excerpt[6])

    return processed_excerpts

def extract_url_and_title(excerpts):
    results = []
    url_pattern = r"https://[^ \n]*?\.html"

    for excerpt in excerpts:
        url_match = re.search(url_pattern, excerpt)
        url = url_match.group(0) if url_match else ""

        # Remove URL from excerpt and clean the rest
        title_raw = excerpt.replace(url, "").strip()
        title_cleaned = re.sub(r"[^a-zA-Z0-9, ]", "", title_raw)

        results.append({
            "url": url,
            "title": title_cleaned
        })

    return results


if __name__ == "__main__":
    extracted_info = []
    for i in tqdm(range(1, 41), desc="Processing pages"):
        html_file_path = f"html_news_datesort/page{i}.html"
        cs_filtered_excerpts = extract_cs_excerpts(html_file_path)
        extracted_info.extend(extract_url_and_title(cs_filtered_excerpts))

    # Build list of instances with id, news_url, news_title
    instances = []
    for i, item in enumerate(extracted_info):
        instances.append({
            "id": i,
            "news_url": item["url"],
            "news_title": item["title"]
        })

    # Save as JSON
    with open("cs_news_urls.json", "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(instances)} items to 'cs_news_urls.json'")