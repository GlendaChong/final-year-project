import os, json, logging, asyncio, pandas as pd
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx
from rank_bm25 import BM25Okapi


load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("OPENROUTER_API_KEY")
)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
JUDGE_MODEL = "deepseek/deepseek-r1" 

sem = asyncio.Semaphore(5)
file_lock = asyncio.Lock()


# COMPONENT-WISE ABLATION MATRIX
ABLATION_MODES = {
    # Discovery Loop Ablation: Searches only ONCE, no pivoting allowed.
    "no_discovery_loop":    {"use_rag": True, "use_full_scrape": True, "use_rubrics": True, "use_judge": True, "use_search_loop": False},
    
    # Drafting Loop Ablation: Drafts only ONCE, no judge critique allowed.
    "no_drafting":          {"use_rag": True, "use_full_scrape": True, "use_rubrics": True, "use_judge": False, "use_search_loop": True},
    
    # Rubric Ablation: Uses generic prompt instead of Connection Rubrics.
    "no_connection_rubrics": {"use_rag": True, "use_full_scrape": True, "use_rubrics": False, "use_judge": True, "use_search_loop": True},
    
    # Scraping Ablation: Uses only the search snippet.
    "no_full_scrape":       {"use_rag": True, "use_full_scrape": False, "use_rubrics": True, "use_judge": True, "use_search_loop": True},

    "full_pipeline":        {"use_rag": True, "use_full_scrape": True, "use_rubrics": True, "use_judge": True, "use_search_loop": True},
}

class RubricLedger:
    def __init__(self):
        self.search_rubrics = []      
        self.anchor_event = "Use general summary mode rubrics (no specific anchor event found)."      
        self.corrective_rubrics = []  
        self.technical_anchor = ""
        self.raw_search_results = []  
        self.judge_rationales = []
        self.history_log = []


class RetrievalStatus:
    SNIPPET_ONLY = "snippet_only"
    PROXY_OK = "proxy_ok"
    PROXY_BLOCKED = "proxy_blocked"
    FALLBACK_SUMMARY = "fallback_summary"


async def serper_search_impact(query, num_results=5):
    url = "https://google.serper.dev/search"
    excluded_domains = ["arxiv.org", "researchgate.net", "scholar.google.com", "nature.com", "ieee.org"]
    core_exclusions = " ".join([f"-site:{d}" for d in excluded_domains])
    payload = {"q": f"{query} {core_exclusions}", "num": 10}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(url, headers=headers, json=payload, timeout=10.0)
            return response.json().get('organic', [])[:num_results]
        except: return []


async def fetch_full_text_with_status(url):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"https://r.jina.ai/{url}", timeout=15)

        text = r.text.lower()

        block_signals = [
            "captcha",
            "security check",
            "cloudflare",
            "access denied",
            "forbidden",
            "enable javascript", 
            "internal server error", 
            "error 500", 
            "404 not found",
            "markdown content:\n\n\n" # Catching the empty markdown skeleton
        ]

        if r.status_code != 200 or any(s in text for s in block_signals):
            return None, RetrievalStatus.PROXY_BLOCKED
        
        # If the page has too many links relative to its length, it's likely a menu or landing page.
        link_count = len(re.findall(r'\[.*?\]\(.*?\)', text))
        if len(text) > 0 and (link_count / (len(text) / 100)) > 5: # Threshold: more than 5 links per 100 chars
             return None, "CONTENT_IS_LINK_LIST"

        return r.text, RetrievalStatus.PROXY_OK

    except Exception:
        return None, RetrievalStatus.PROXY_BLOCKED


def get_newsworthy_chunks(full_text, query):
    paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
    if not paragraphs: return ""
    tokenized_corpus = [p.lower().split() for p in paragraphs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
    return "\n\n".join([c[0] for c in ranked[:3]])


def safe_parse_json(content):
    try:
        clean_content = re.sub(r'```json|```', '', content).strip()
        return json.loads(clean_content)
    except:
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        return json.loads(match.group(1)) if match else None


async def phase_1_init(abstract, introduction, model_path):
    prompt = f"""
    [Research Paper Abstract]: {abstract}
    [Research Paper Introduction]: {introduction}
    
    Task:
    Using the research paper abstract and introduction, generate: 
    1. Technical Anchor: A concise statement of the core technical contribution or theme of the paper that can serve as the central 'hook' for newsworthiness. This should be a specific aspect of the research that has clear real-world implications or connections to current industry trends.
    2. Narrative Search Query: Create a SHORT (4-7 words) query using industry keywords relevant to the paper.
       - Prefer search query terms that imply concrete events or news. 
       - Not a generic query/trend/background term. 

    MUST Return JSON ONLY: {{"search_query": "string", "technical_anchor": "string"}}
    """
    res = await client.chat.completions.create(model=model_path, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)


async def phase_2_discovery_loop(title, abstract, intro, initial_query, ledger, model_path, use_loop=True, use_scrape=True):
    current_query = initial_query
    
    # Ablation condition: max_attempts is 5 for Full Pipeline, but 1 for 'no_discovery_loop'
    max_attempts = 5 if use_loop else 1
    
    for attempt in range(max_attempts):
        results = await serper_search_impact(current_query)
        if not results:
            # If no results at all, we pivot once even in "no_loop" to avoid total failure
            current_query = f"{title} industry trends"
            continue

        snippets_text = "\n".join([f"[{i}] {r.get('title')}: {r.get('snippet')}" for i, r in enumerate(results)])
        
        judge_prompt = f"""
        Primary Research: {abstract} and {intro}
        Search Results: {snippets_text}
        Technical Anchor: {ledger.technical_anchor}
        Search Query Used: {current_query}

        Task: 
        1. Identify search results that's highly relevant to the primary research. 
        2. If no suitable anchor event is found, generate a pivot query. 

        Return JSON ONLY: {{
            "found": bool, 
            "anchor_details": "string", 
            "relevant_indices": [int],
            "pivot_query": "string"
        }}
        """
        res = await client.chat.completions.create(
            model=model_path, 
            messages=[{"role": "user", "content": judge_prompt}], 
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        
        if data['found'] and results:
            best_idx = data['relevant_indices'][0] if data['relevant_indices'] else 0
            best_result = results[best_idx]
            
            # ABLATION CONDITION: Scraping vs Snippet only
            if use_scrape:
                text, status = await fetch_full_text_with_status(best_result['link'])
                # If scraping fails (proxy_blocked), we fallback to snippet to save the score
                if status == "proxy_ok" and text:
                    content = get_newsworthy_chunks(text, ledger.technical_anchor)
                else:
                    content = best_result['snippet'] # Robust Fallback
            else:
                content = best_result['snippet']
            
            ledger.anchor_event = {
                "summary": data['anchor_details'], 
                "full_content": content, 
                "source_url": best_result['link']
            }
            return content
        
        # Update query for next loop iteration
        current_query = data.get('pivot_query', f"{title} industry news")
        
    return "No anchor found."



async def phase_2_pivot_rubrics(ledger, model_path):
    pivot_prompt = f"""
    Technical Anchor: {ledger.technical_anchor}
    Retrieved Content: {ledger.anchor_event.get('full_content', '')}
    
    Task: Create two 'Connection Rubrics' for the writer and judge using the retrieved content to ensure the final article effectively links the research to the retrieved content. 

    STRICT GROUNDING RULES:
    1. Every specific detail in the rubrics MUST exist in the retrieved content. Do not fabricate or assume any details that are not explicitly stated in the retrieved content.
    2. DO NOT use "e.g." or "for example" to suggest data points that are not in the snippets.

    Rubrics:
    1. 'Hook' constraint: How the article must being using the retrieved content to hook readers, ensuring it is directly tied to the technical anchor. 
    2. 'Bridge' constraint: Logical link between the technical anchor and retrieved content. This should guide how the article connects the real-world event to the research findings for a coherent narrative. 

    MUST Return JSON ONLY: {{"hook_rubric": "string", "bridge_rubric": "string"}}
    """
    res = await client.chat.completions.create(model=model_path, messages=[{"role": "user", "content": pivot_prompt}], response_format={"type": "json_object"})
    data = safe_parse_json(res.choices[0].message.content)
    ledger.corrective_rubrics.append(f"HOOK CONSTRAINT: {data.get('hook_rubric')}")
    ledger.corrective_rubrics.append(f"BRIDGE CONSTRAINT: {data.get('bridge_rubric')}")



async def phase_3_drafting_loop(row, ledger, model_path, use_judge=True):
    abstract, intro = row.get('Abstract', ''), row.get('Introduction', '')
    citation = row.get('citation', '')
    
    # Ablation condition: 5 rounds vs 1 round
    rounds = 5 if use_judge else 1
    
    for attempt in range(rounds):
        active_rubrics = "\n".join([f"- {r}" for r in ledger.corrective_rubrics])
        
        writer_prompt = f"""
        Persona: Experienced computer science journalist. Audience is expert readers with computer science research backgrounds.
        Core Perspective: Your voice is sophisticated, analytical, and professional. You must prioritize technical significance, expert-level implications, and nuanced breakthroughs.

        TASK: Write/Refine a 600-900 word news article using these sources: 
        
        SOURCES:
        [Abstract]: {abstract}
        [Introduction]: {intro}
        [Citation]: {citation}
        [Real-World Anchor Event]: {ledger.anchor_event}

        EVOLVING RUBRICS (CONSTRAINTS TO SATISFY):
        {active_rubrics}

        STRICT EDITORIAL GUIDELINES:
        1. SOURCE HIERARCHY: Every technical claim must be verifiable against the Abstract, Introduction and Citation, and should be the main technical contribution of the paper.
        2. NO DESCRIPTIVE INFERENCE: Do not add qualitative adjectives or industry status descriptors unless explicitly present in sources.
        3. NO QUANTITATIVE ASSUMPTIONS: Do not estimate or number processes unless explicitly stated.
        4. ANCHOR INTEGRATION: Use the Anchor Event in the first paragraph to provide a real-world hook.
        5. TECHNICAL TERMINOLOGY: Use exact terminology from the paper.
        6. STRUCTURE: Follow the Inverted Pyramid.
        7. Conclude with a motivated Call to Action to access the research paper.

        Format:
        Headline: [Headline]
        News Article: [Content]
        """
        res = await client.chat.completions.create(model=model_path, messages=[{"role": "user", "content": writer_prompt}])
        draft = res.choices[0].message.content

        # ABLATION CONDITION: Skip judge if False
        if not use_judge: return draft

        judge_prompt = f"""
        ROLE: Adversarial Technical Editor.
        [Target Rubrics to Satisfy]: {active_rubrics}
        [Draft]: {draft}
        [Source Abstract]: {abstract}
        [Technical Anchor]: {ledger.technical_anchor}

        TASK: 
        1. Evaluate if the [Draft] satisfies the [Target Rubrics].
        2. Ensure technical fidelity.

        EVOLUTION RULE: 
        If the draft fails, mark 'pass': false. Generate a new 'Instance-Specific Corrective Rubric'.

        OUTPUT JSON ONLY:
        {{ "pass": bool, "rationale": "string", "corrective_rubric": "string" }}
        """
        eval_res = await client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": judge_prompt}], response_format={"type": "json_object"})
        feedback = safe_parse_json(eval_res.choices[0].message.content)
        
        if not feedback or feedback.get('pass'): return draft
        ledger.corrective_rubrics.append(feedback.get('corrective_rubric'))
        ledger.judge_rationales.append(feedback.get('rationale'))
    return draft


async def process_paper_ablation(idx, row, mode_alias, model_path, output_path, config):
    async with sem:
        paper_title = row.get('news_title', 'Unknown')
        ledger = RubricLedger()
        try:
            # Phase 1: Init
            init_data = await phase_1_init(row['Abstract'], row.get('Introduction',''), model_path)
            ledger.technical_anchor = init_data['technical_anchor']
            
            # Phase 2: Discovery Loop with optional full scraping
            # This handles both "no_discovery_loop" (max_attempts=1) and "no_full_scrape"
            await phase_2_discovery_loop(
                paper_title, row['Abstract'], row.get('Introduction',''), 
                init_data['search_query'], ledger, model_path, 
                use_loop=config.get("use_search_loop", True), 
                use_scrape=config.get("use_full_scrape", True)
            )
            
            # Phase 2b: Rubric pivot
            if config.get("use_rubrics", True):
                await phase_2_pivot_rubrics(ledger, model_path)
            else:
                ledger.corrective_rubrics = ["Write a professional CS news article based on the provided research and anchor."]

            # Phase 3: Drafting 
            final_article = await phase_3_drafting_loop(
                row, ledger, model_path, 
                use_judge=config.get("use_judge", True)
            )

            # Save
            result_payload = {
                "row_index": idx, 
                "generation_model": "openai/gpt-4.1", 
                "ablation_mode": mode_alias,
                "title": paper_title,
                "abstract": row['Abstract'],
                "introduction": row.get('Introduction', ''),
                "citation": row.get('citation', ''),
                "rag_generated_news_article": final_article, 
                "ledger": vars(ledger), 
            }

            await save_result(result_payload, output_path)
            logging.info(f"✅ [{mode_alias}] {idx} Complete")
        except Exception as e:
            logging.error(f"Error on {idx}: {e}")


async def save_result(result, file_path):
    async with file_lock:
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f: data = json.load(f)
        data.append(result)
        with open(file_path, 'w') as f: json.dump(data, f, indent=4)


async def main():
    os.makedirs('ablation_results', exist_ok=True)
    # Load the full source data
    full_df = pd.read_json('extracted_papers_summary_5.json').head(50)
    target_model = "openai/gpt-4.1"

    for mode_name, config in ABLATION_MODES.items():
        output_file = f"ablation_results/results_{mode_name}.json"
        
        # Identify which indices are already finished
        finished_indices = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    finished_indices = {item['row_index'] for item in existing_data}
            except Exception as e:
                logging.error(f"Error reading {output_file}: {e}")

        # Filter the dataframe to only include indices not in finished_indices
        missing_df = full_df[~full_df.index.isin(finished_indices)]
        
        if missing_df.empty:
            logging.info(f"✅ Mode {mode_name.upper()} is already complete (50/50).")
            continue

        logging.info(f"🚀 RESUMING MODE: {mode_name.upper()} | Missing Indices: {list(missing_df.index)}")
        
        # Create tasks only for the missing rows
        tasks = [
            process_paper_ablation(i, row, mode_name, target_model, output_file, config) 
            for i, row in missing_df.iterrows()
        ]
        
        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())