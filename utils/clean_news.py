import pandas as pd
import sys, os
import json
import re, html
from tqdm import tqdm
import shutil
sys.path.append(os.getcwd())



# Menu/section crumbs to drop when they appear as standalone lines
STOPLINES = {
    "topics", "science x account", "discover more", "share this!",
    "home", "the gist", "editors' notes", "sign in", "sign up",
    "forget password?", "not a member? sign up", "learn more",
    "automotive", "business", "computer sciences", "consumer & gadgets",
    "electronics & semiconductors", "energy & green tech", "engineering",
    "hardware", "hi tech & innovation", "internet", "machine learning & ai",
    "other", "robotics", "security", "software", "telecom",
    "electronics", "science", "sciences", "artificial intelligence",
    "tweet", "share", "email"
}

def _looks_like_menu(line: str) -> bool:
    low = line.lower().strip(" -*•\t")
    if low in STOPLINES:
        return True
    # numbered breadcrumbs and lone digits/bullets
    if "credit:" in low:
        return True
    if re.fullmatch(r"\d+\.\s*[a-z &]+", low):
        return True
    # blank link remnants like "[](" or "[])"
    if re.fullmatch(r"\[\]\(?\)?", line.strip()):
        return True
    # lines that are just "[](" or "](" or "[]"
    if any(tok in line for tok in ("[](", "](", "[])")) and not re.search(r"\w", line):
        return True
    # single symbol or tiny lines with no letters
    if len(line.strip()) < 3 or not re.search(r"[A-Za-z]", line):
        return True
    return False

def _looks_like_prose(line: str) -> bool:
    s = line.strip()
    # keep lines with letters and either: long-ish or sentence punctuation
    if not re.search(r"[A-Za-z]", s):
        return False
    if len(s) >= 60:
        return True
    if re.search(r"[\.!?][\"”’)]?\s*$", s):
        return True
    # short but likely a paragraph continuation
    if len(s) >= 35 and " " in s:
        return True
    return False

def clean_article_only(text: str, stop_at_citation: bool = True) -> str:
    # normalize & unescape
    s = html.unescape(text.replace("\r\n", "\n").replace("\r", "\n"))

    # remove markdown/HTML images and figures
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)                 # ![alt](url)
    s = re.sub(r"<img\b[^>]*>", "", s, flags=re.I)
    s = re.sub(r"</?(figure|figcaption)\b[^>]*>", "", s, flags=re.I)

    # convert [text](url) -> text ; remove bare URLs
    s = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)

    # drop markdown headers/separators/backticks
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", s)
    s = re.sub(r"^\s*(?:-{3,}|\*{3,}|_{3,}|=+)\s*$", "", s, flags=re.M)

    # remove bracketed image/aside labels like [Image 5: ...]
    s = re.sub(r"\[\s*Image[^]]*\]", "", s, flags=re.I)
    s = re.sub(r"\(\s*The GIST\s*\)\)?", "", s, flags=re.I)

    # optionally cut off before Citation
    if stop_at_citation:
        m = re.search(r"^\s*\*{0,2}Citation\*{0,2}\s*:", s, flags=re.I | re.M)
        if m:
            s = s[:m.start()]

    # line-level filtering
    lines = []
    for raw in s.split("\n"):
        line = raw.strip()
        if not line:
            lines.append("")  # preserve paragraph breaks
            continue
        # kill leftover empty link stubs like "[](" blocks
        if _looks_like_menu(line):
            continue
        # drop lines that are mostly punctuation/brackets
        if re.fullmatch(r"[\[\]()/|:;.,*'\"` -]+", line):
            continue
        lines.append(line)

    # now keep ONLY the largest contiguous block of prose-like lines
    blocks, cur = [], []
    def flush():
        if cur:
            blocks.append(cur[:])
            cur.clear()
    for line in lines:
        if not line:
            # paragraph boundary counts as part of a prose block if we’re inside one
            cur.append("")
        elif _looks_like_prose(line):
            cur.append(line)
        else:
            flush()
    flush()

    if not blocks:
        return ""

    # score blocks by character count
    best = max(blocks, key=lambda b: sum(len(x) for x in b))
    out = "\n".join(best)

    # tidy: collapse 3+ blanks → 1; strip stray bullets at starts of lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"^[\*\-•]\s*", "", out, flags=re.M)
    return out.strip().replace("_", "")


def extract_title(lines):

    for i, line in enumerate(lines):
        
        if line[:6] == "Title:":
            title = line.split("Title:")[1].strip()
            
            return title

    return None

def extract_doi_link(lines):

    for i, line in enumerate(lines):
        
        if "dx.doi.org" in line:
            start = line.rfind("https://")
            
            return line[start:].replace(")\n", "")
    return None

def extract_excerpt(lines):
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if start_idx is None and " Credit: " in line:
            start_idx = i + 1  # start after this line
        elif start_idx is None and "Markdown Content:" in line:
            start_idx = i + 1  # start after this line
        elif start_idx is not None and "**More information:**" in line:
            end_idx = i
            break
        elif start_idx is not None and end_idx is None and "Provided by" in line:
            end_idx = i
            break
        elif start_idx is not None and end_idx is None and "republished from" in line:
            end_idx = i
            break
        elif start_idx is not None and end_idx is None and "Citation" in line:
            end_idx = i
            break
    if start_idx is not None and end_idx is not None:
        return ('').join(lines[start_idx:end_idx])
    else:
        print("Excerpt markers not found", start_idx, end_idx)


if __name__ == "__main__":
    extracted_info = []
    citations = []
    sanitized_ids = []
    sanitized_json = []

    # Load your source list
    all_cs_news = json.load(open("cs_news_urls.json", "r", encoding="utf-8"))

    # --- STEP 1: SANITY CHECK (Modified) ---
    for i in tqdm(range(len(all_cs_news)), desc="Processing pages"):
        file_path = f"techxplore_news_pages/{i}.html"
        
        # Safety check: skip if the file was never downloaded
        if not os.path.exists(file_path):
            citations.append(None)
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        title = extract_title(lines)
        citation = extract_doi_link(lines) # This can return None
        
        citations.append(citation)

        # citation is allowed to be None.
        if title is not None:
            sanitized_ids.append(i)
    
    print(f"{len(sanitized_ids)} out of {len(all_cs_news)} passed sanitation check.")

    # --- STEP 2: RE-PROCESSING ---
    src_dir = "techxplore_news_pages"
    dst_dir = "techxplore_news_pages_sanitized"
    os.makedirs(dst_dir, exist_ok=True)

    for new_id, old_id in enumerate(tqdm(sanitized_ids, desc="Re-processing sanitized pages")):
        src_path = f"{src_dir}/{old_id}.html"
        dst_path = f"{dst_dir}/{new_id}.html"
        
        shutil.copyfile(src_path, dst_path)
        
        json_item = all_cs_news[old_id]

        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        article = extract_excerpt(lines)
        
        # Ensure clean_text doesn't crash if excerpt markers weren't found
        clean_text = clean_article_only(article) if article else ""
        
        sanitized_json.append({
            "id": new_id,
            "news_title": json_item["news_title"],
            "news_url": json_item["news_url"],
            "citation": citations[old_id], 
            "news_article": clean_text
        })

    with open("sanitized_cs_news_data.json", "w", encoding="utf-8") as f:
        json.dump(sanitized_json, f, indent=2, ensure_ascii=False)