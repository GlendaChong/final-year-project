import streamlit as st
import fitz  # PyMuPDF
import json
import os
import re
import asyncio
import requests
from openai import AsyncOpenAI
from rag_final.generate_rag_evolving_rubric import main as run_rag_engine, EXPERIMENT_MODELS


DEMO_DIR = 'demo'
ARCHIVE_DIR = 'rag_final'
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
api_key = os.getenv("OPENROUTER_API_KEY")

# Centered layout for the "Nature Journal" longform aesthetic
st.set_page_config(page_title="The Scientific Gazette", layout="centered", page_icon="🗞️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Source+Serif+Pro:ital,wght@0,400;0,700;1,400&display=swap');
    
    /* 1. TYPOGRAPHY (Uses inherited theme colors) */
    .news-header { 
        font-family: 'Playfair Display', serif; font-size: clamp(40px, 8vw, 72px); 
        text-align: center; border-bottom: 4px double currentColor; 
        margin-top: 20px; padding-bottom: 10px;
    }
    
    .date-header {
        text-align: center; font-family: 'Source Serif Pro', serif; 
        text-transform: uppercase; letter-spacing: 4px; font-size: 14px; 
        margin-bottom: 40px; border-bottom: 2px solid currentColor; padding-bottom: 10px;
        opacity: 0.8;
    }

    .article-title { 
        font-family: 'Playfair Display', serif; font-size: clamp(32px, 5vw, 52px); 
        font-weight: bold; line-height: 1.1; margin-bottom: 25px; text-align: left;
    }

    .article-content { 
        font-family: 'Source Serif Pro', serif; font-size: 22px; 
        line-height: 1.8; text-align: justify; 
    }
            
    /* Traditional Drop Cap - Forced to stay bold */
    .article-content::first-letter {
        float: left; font-size: 85px; line-height: 70px; padding-top: 8px;
        padding-right: 12px; padding-left: 3px; font-family: 'Playfair Display', serif;
        font-weight: bold;
    }

    /* 2. COMPONENT STYLING */
    .resource-box {
        border-top: 2px solid currentColor; border-bottom: 2px solid currentColor;
        padding: 30px; margin-top: 50px; font-family: 'Source Serif Pro', serif; 
        font-style: italic; font-size: 18px; opacity: 0.9;
    }
    
    /* Ensure the button is big and readable */
    .stButton > button {
        width: 100% !important;
        font-size: 20px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }

    /* Remove the 'Force White' background and 'Force Black' text logic */
    </style>
    """, unsafe_allow_html=True)



def extract_layout_aware_text(pdf_file):
    """Extracts text from the first few pages of a PDF while preserving the left-to-right reading order, even in multi-column layouts."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        if page.number > 3: break
        blocks = page.get_text("blocks")
        midpoint = page.rect.width / 2
        blocks.sort(key=lambda b: (0 if b[0] < (midpoint + 20) else 1, b[1]))
        for b in blocks:
            if b[6] == 0: full_text += b[4] + "\n"
    return full_text


def parse_pdf_sections(text):
    """A heuristic-based parser to extract Abstract and Introduction sections from the raw text of a research paper."""
    res = {"Abstract": "", "Introduction": ""}
    abs_m = re.search(r"(?i)\bABSTRACT\b", text)
    intro_m = re.search(r"(?i)\n(?:1[\.\s]*)?INTRODUCTION", text)
    if abs_m and intro_m:
        res["Abstract"] = text[abs_m.end():intro_m.start()].strip()
        next_sec = re.search(r"(?i)\n(?:2[\.\s]*)(?:RELATED WORK|BACKGROUND|METHOD)", text[intro_m.end():])
        end_idx = intro_m.end() + next_sec.start() if next_sec else intro_m.end() + 4000
        res["Introduction"] = text[intro_m.end():end_idx].strip()
    return res


def parse_and_clean_links(text):
    """Converts Markdown-style links and raw URLs into styled HTML anchor tags, ensuring they open in new tabs and are visually distinct."""
    # 1. Handle Markdown-style links: [Text](URL)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', 
                  r'<a href="\2" target="_blank" style="color: #007bff; text-decoration: underline;">\1</a>', text)
    
    # 2. Handle raw URLs (lookbehind ensures we don't double-process hrefs from step 1)
    # This regex captures most modern URL structures including those in your screenshot
    raw_url_pattern = r'(?<!href=")(https?://[a-zA-Z0-9\.\/\-\_\?\&\=\#\%]+)'
    text = re.sub(raw_url_pattern, 
                  r'<a href="\1" target="_blank" style="color: #007bff; text-decoration: underline;">\1</a>', text)
    return text


def get_openrouter_image(technical_anchor, api_key):
    """Generates an image using OpenRouter's 2026 multimodal endpoint."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-2.5-flash-image", 
        "messages": [
            {
                "role": "user", 
                "content": f"A clean, professional editorial stock photo for a scientific news articles. Subject: {technical_anchor}. Technical, high-resolution, minimalist."
            }
        ],
        "modalities": ["image", "text"], 
        "image_config": {
            "aspect_ratio": "16:9"  # Widescreen format ideal for news articles
        }   
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        result = response.json()
        
        # Check for API-level errors
        if "error" in result:
            error_msg = result["error"].get("message", "Unknown Error")
            st.error(f"OpenRouter Error: {error_msg}")
            return None

        # Extract according to documentation snippet
        if result.get("choices"):
            message = result["choices"][0]["message"]
            if message.get("images"):
                image_data = message["images"][0] # Take the first image if multiple are returned
                return image_data["image_url"]["url"]
        
        return None

    except Exception as e:
        print(f"Function Exception: {e}")
        return None

def render_article(art_obj):
    """Clean, single-column rendering for expert journalism without technical metadata."""
    raw = art_obj.get("rag_generated_news_article", "")
    
    # Extract Clean Headline
    headline = art_obj.get('title', 'Scientific Discovery')
    if "Headline:" in raw:
        headline = raw.split("Headline:")[1].split("News Article:")[0].strip()
    
    # Extract Clean Body
    body = raw
    if "News Article:" in raw:
        body = raw.split("News Article:")[1].split("End of News Article.")[0].strip()

    # Separate Technical Exposition Footer
    expo_marker = "For a comprehensive technical exposition"
    main_body, expo_text = (body.split(expo_marker)[0], expo_marker + body.split(expo_marker)[1]) if expo_marker in body else (body, "")

    # UI Display
    st.markdown(f"<div class='article-title'>{headline}</div>", unsafe_allow_html=True)
    
    # Image Handling
    image_url = art_obj.get("metadata", {}).get("image_url")
    anchor = art_obj.get('ledger', {}).get('technical_anchor', 'Scientific Research')

    if not image_url:
        # Try to generate one on the fly
        with st.spinner(f"Generating visual for '{anchor}'..."):
            generated_url = get_openrouter_image(anchor, api_key)
            if generated_url:
                image_url = generated_url

    # Final Display
    if image_url:
        st.image(image_url, width='stretch', caption=f"Visualizing: {anchor}")
    else:
        st.warning("Could not load image.")

    # Process main news article body
    processed_body = parse_and_clean_links(main_body).replace("\n", "<br>")
    st.markdown(f"<div class='article-content'>{processed_body}</div>", unsafe_allow_html=True)

    # Technical Footer
    if expo_text:
        st.markdown(f"<div class='resource-box'>{parse_and_clean_links(expo_text)}</div>", unsafe_allow_html=True)

    # Technical Expander (Hidden from main reading view)
    st.divider()
    with st.expander("🛠️ View Agentic Trace & Evolved Rubrics"):
        st.write(f"**Generation Model:** {art_obj.get('generation_model')}")
        st.write(f"**Paper Index:** {art_obj.get('row_index')}")
        st.subheader("🕵️ Agentic History")
        st.json(art_obj.get('ledger', {}).get('history_log', []))
        st.subheader("🎯 Evolved Rubrics")
        st.json(art_obj.get('ledger', {}).get('corrective_rubrics', []))


# Main website logic
st.markdown("<div class='news-header'>THE SCIENTIFIC GAZETTE</div>", unsafe_allow_html=True)
st.markdown("<div class='date-header'>SINGAPORE | SUNDAY, APRIL 19, 2026</div>", unsafe_allow_html=True)

mode = st.sidebar.radio("Navigation", ["Live Demo Pipeline", "Archive Viewer"])

if mode == "Archive Viewer":
    # Archive Viewer: Browse previously generated articles with search and filter capabilities
    json_path = os.path.join(ARCHIVE_DIR, 'rag_generated_articles_evolving_rubric_full_gpt4_1.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        search = st.sidebar.text_input("🔍 Search Archive", "")
        filtered = [a for a in data if search.lower() in a.get('title', '').lower()] if search else data
        if filtered:
            selected_title = st.sidebar.selectbox("Select Article", [a.get('title') for a in filtered])
            render_article(next(a for a in filtered if a.get('title') == selected_title))
    else:
        st.error("Archive not found.")

else:
    # Live Demo Pipeline
    input_method = st.radio("Input Source", ["Auto-Scrape PDF", "Manual Entry"], horizontal=True)
    
    abs_in, intro_in, p_name = "", "", "Research Paper"

    if input_method == "Auto-Scrape PDF":
        uploaded_file = st.file_uploader("Upload Research Paper PDF", type="pdf")
        if uploaded_file:
            p_name = uploaded_file.name.replace(".pdf", "")
            with st.spinner("Decoding layout..."):
                raw = extract_layout_aware_text(uploaded_file)
                secs = parse_pdf_sections(raw)
                abs_in = st.text_area("Verify Abstract", secs["Abstract"], height=150)
                intro_in = st.text_area("Verify Introduction", secs["Introduction"], height=200)
                citation_val = st.text_input("Citation", f"Source: {p_name}")
    else:
        p_name = st.text_input("Paper Title", "New Discovery")
        abs_in = st.text_area("Paste Abstract", height=150)
        intro_in = st.text_area("Paste Introduction", height=200)
        citation_val = st.text_input("Citation", f"Source: {p_name}")


    if st.button("🚀 Generate Journalism"):
        if abs_in and intro_in:
            live_input_path = os.path.join(DEMO_DIR, 'live_upload_input.json')
            with open(live_input_path, 'w') as f:
                json.dump([{"row_index": 999, "news_title": p_name, "Abstract": abs_in, "Introduction": intro_in, "citation": p_name}], f, indent=4)
            
            with st.status("Agentic Drafting...", expanded=False) as s:
                try:
                    # Run RAG Engine
                    asyncio.run(run_rag_engine(file_path=live_input_path, model_file_prefix="demo/demo_output_v2"))
                    
                    # Add AI Image Generation to the output metadata
                    res_file = os.path.join(DEMO_DIR, "demo_output_v2_gpt4_1.json")
                    if os.path.exists(res_file):
                        with open(res_file, 'r') as f:
                            res_data = json.load(f)
                        
                        anchor = res_data[0].get('ledger', {}).get('technical_anchor', 'Science')
                        img_url = get_openrouter_image(anchor, api_key)
                        
                        if "metadata" not in res_data[0]: res_data[0]["metadata"] = {}
                        res_data[0]["metadata"]["image_url"] = img_url
                        
                        with open(res_file, 'w') as f:
                            json.dump(res_data, f, indent=4)

                    s.update(label="✅ Success!", state="complete")
                    st.divider()
                    
                    # Render the newly created article
                    render_article(res_data[0])

                except Exception as e:
                    st.error(f"Engine Error: {e}")