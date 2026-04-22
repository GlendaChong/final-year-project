import os, json, logging, asyncio, pandas as pd
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx
from rank_bm25 import BM25Okapi

# --- INITIALIZATION ---
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("OPENROUTER_API_KEY")
)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# JUDGE_MODEL = "anthropic/claude-3.5-sonnet"
JUDGE_MODEL = "deepseek/deepseek-r1-0528"

sem = asyncio.Semaphore(5)
# OUTPUT_PATH = 'rag_final/rag_generated_articles_evolving_rubric_full.json'
file_lock = asyncio.Lock()


EXPERIMENT_MODELS = {
    "gpt4_1": "openai/gpt-4.1",
    "gpt4o": "openai/gpt-4o",
    "gemini2_5_flash": "google/gemini-2.5-flash",
    "llama_3_1_70b": "meta-llama/llama-3.1-70b-instruct",
    "llama_3_1_8b": "meta-llama/llama-3.1-8b-instruct",
    "deepseek_v3": "deepseek/deepseek-chat",
    "deepseek_r1": "deepseek/deepseek-r1-distill-qwen-32b"
}


class RubricLedger:
    def __init__(self):
        self.search_rubrics = []      
        self.anchor_event = "Use general summary mode rubrics (no specific anchor event found)."      
        self.corrective_rubrics = []  
        self.technical_anchor = ""
        self.raw_search_results = []  # Store raw search results for potential later use
        self.judge_rationales = []
        self.history_log = []  # To store: {mode, round, action, rationale}

class RetrievalStatus:
    SNIPPET_ONLY = "snippet_only"
    PROXY_OK = "proxy_ok"
    PROXY_BLOCKED = "proxy_blocked"
    FALLBACK_SUMMARY = "fallback_summary"


# --- UTILITIES ---
async def serper_search_impact(query, num_results=5):
    url = "https://google.serper.dev/search"
    core_exclusions = "-site:arxiv.org -site:researchgate.net -site:springer.com -site:sciencedirect.com -site:ieee.org -site:nature.com -site:openreviews.net -site:neurips.cc -site:icml.cc -site:cv-foundation.org -site:scholar.google.com -site:openreviews.net"
    social_exclusions = "-site:linkedin.com -site:twitter.com -site:facebook.com -site:medium.com" 
    final_q = f"{query} {core_exclusions} {social_exclusions} "

    payload = {
        "q": final_q,
        "num": 10, 
    }
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(url, headers=headers, json=payload, timeout=10.0)
            results = response.json().get('organic', [])
            return results[:num_results]
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []


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
        if len(text) > 0 and (link_count / (len(text) / 100)) > 5: 
             return None, "CONTENT_IS_LINK_LIST"

        return r.text, RetrievalStatus.PROXY_OK

    except Exception:
        return None, RetrievalStatus.PROXY_BLOCKED


def get_newsworthy_chunks(full_text, query, chunk_size=600):
    """
    Refined IR: Prioritizes chunks that contain 'News Signals' 
    (numbers, entities, or recent dates) alongside BM25 relevance.
    """
    paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
    
    if not paragraphs: return ""

    # 1. BM25 Scoring
    tokenized_corpus = [p.lower().split() for p in paragraphs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    # 2. News Signal Heuristics (Boost chunks with data)
    final_ranked_chunks = []
    for i, p in enumerate(paragraphs):
        heuristic_score = scores[i]
        if re.search(r'\d+%', p): heuristic_score *= 1.5 # Boost percentages
        if re.search(r'\$\d+', p): heuristic_score *= 1.3 # Boost monetary values
        if re.search(r'(202[4-6])', p): heuristic_score *= 1.2 # Boost recent years
        
        final_ranked_chunks.append((p, heuristic_score))

    # Sort by boosted score
    final_ranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 chunks to provide a broader 'Bridge'
    return "\n\n[---]\n\n".join([c[0] for c in final_ranked_chunks[:3]])


def safe_parse_json(content):
    try:
        # Remove markdown code blocks if the model included them
        clean_content = re.sub(r'```json|```', '', content).strip()
        return json.loads(clean_content)
    except json.JSONDecodeError:
        # If it still fails, try to find anything between curly braces
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return None



# --- PHASE 1: INITIALIZATION & HYPOTHESIS GEN ---
async def phase_1_init(abstract, introduction, model_path):
    # Matches "Analyze Paper -> Generate News Angle Search Rubrics"
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
    res = await client.chat.completions.create(
        model=model_path, 
        messages=[{"role": "user", "content": prompt}], 
        response_format={"type": "json_object"}
    )

    logging.info(f"Phase 1 Output: {res.choices[0].message.content}")
    return json.loads(res.choices[0].message.content)


def extract_rubrics(raw_response):
    """
    Extracts hook_rubric and bridge_rubric from messy LLM strings.
    Handles: Markdown blocks, nested Schema.org 'text' fields, and plain JSON.
    """
    # 1. Try direct JSON load
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # 2. If it fails, search for the first '{' and last '}' using Regex
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # 3. Handle the 'Schema.org' / nested 'text' issue seen in your logs
    if "text" in data and isinstance(data["text"], str):
        # Recursively try to find JSON inside the 'text' string
        return extract_rubrics(data["text"])

    # 4. Final Validation & Fallback
    hook = data.get("hook_rubric")
    bridge = data.get("bridge_rubric")

    if not hook or not bridge:
        logging.warning("Rubrics missing in response. Using placeholders.")
        return {
            "hook_rubric": hook or "Provide a compelling introduction to the research.",
            "bridge_rubric": bridge or "Connect the technical data to real-world impact."
        }

    return {"hook_rubric": hook, "bridge_rubric": bridge}
    

async def phase_2_discovery_and_pivot(title, abstract, introduction, initial_query, ledger, model_path):
    current_query = initial_query
    anchor_found = False
    final_content_for_writer = ""  # This replaces the inconsistent full_scraped_content

    for attempt in range(5):
        if current_query not in ledger.search_rubrics:
            ledger.search_rubrics.append(current_query)

        results = await serper_search_impact(current_query)
        ledger.raw_search_results.append({
            "attempt": attempt + 1, 
            "query": current_query, 
            "results": results
        })

        snippets_text = "\n".join([f"[{i}] {r.get('title', 'No Title')}: {r.get('snippet', 'No snippet available')}" for i, r in enumerate(results)])
        
        judge_prompt = f"""
        Primary Research: {abstract} and {introduction}
        Search Results: {snippets_text}
        Technical Anchor: {ledger.technical_anchor}
        Search Query Used: {current_query}

        Task: 
        1. Identify search results that's highly relevant to the primary research. It should not be the same primary research as the paper. 
        2. Rank relevant indices by technical depth and source quality to serve as a hook for the article.
        3. If no suitable anchor event is found, state the rationale and generate a pivot query of a different angle as the current search query used. 

        Return JSON ONLY: {{
            "found": bool, 
            "anchor_details": "string", 
            "relevant_indices": [list of ints],
            "rationale": "string if found=false",
            "pivot_query": "string"
        }}
        """
        
        res = await client.chat.completions.create(
            model=model_path, 
            messages=[{"role": "user", "content": judge_prompt}], 
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        
        if data['found']:
            search_hits = results
            raw_indices = data.get('relevant_indices', [])
            top_indices: list[int] = []

            for item in raw_indices:
                # Case A: already int
                if isinstance(item, int):
                    top_indices.append(item)
                    continue

                # Case B: list like [0]
                if isinstance(item, list) and item and isinstance(item[0], int):
                    top_indices.append(item[0])
                    continue

                # Case C: dict like {"index": 0}
                if isinstance(item, dict):
                    for key in ("index", "idx", "value"):
                        if key in item and isinstance(item[key], int):
                            top_indices.append(item[key])
                            break
                    continue

                # Case D: string that starts with a number, e.g.
                if isinstance(item, str):
                    m = re.match(r"\s*(\d+)", item)
                    if m:
                        top_indices.append(int(m.group(1)))
                    else:
                        logging.warning(f"Skipping non-numeric index string: {item!r}")
                    continue

                logging.warning(f"Skipping unexpected index format: {item!r} (type={type(item)})")

                    
            # --- ATTEMPT SCRAPING CANDIDATES ---
            for rank, idx in enumerate(top_indices):
                try:
                    if isinstance(idx, list) and len(idx) > 0:
                        idx = int(idx[0])
                    else:
                        idx = int(idx)

                    if 0 <= idx < len(search_hits):

                        current_hit = search_hits[idx]
                        target_url = current_hit.get('link', '')  # Safer access to 'link'
                        
                        print(f"🔄 Round {attempt+1}, Scrape {rank+1}: {target_url}")
                        
                        scraped_text, status = await fetch_full_text_with_status(target_url)
                        
                        if status == RetrievalStatus.PROXY_OK:
                            print(f"✅ Full Scrape Success [{target_url}]: {results[idx]['title']}")
                            
                            final_content_for_writer = get_newsworthy_chunks(scraped_text, ledger.technical_anchor)

                            ledger.anchor_event = {
                                "summary": data['anchor_details'],
                                "snippet": current_hit.get('snippet', ''),
                                "full_content": final_content_for_writer,
                                "source_url": target_url,
                                "mode": "full_scrape"
                            }
                            
                            anchor_found = True
                            break 
                except (IndexError, KeyError): continue

            if anchor_found:
                break 
            
            # If it's the last round and all scrapes in this round failed
            if attempt == 4:
                print("🚨 Max attempts reached. Promoting snippet to full_content.")
                best_idx = top_indices[0] if top_indices else 0
                snippet_text = results[best_idx]['snippet'] if results else "No context found."
                ledger.anchor_event = {
                    "summary": data['anchor_details'],
                    "snippet": snippet_text,
                    "full_content": snippet_text, 
                    "source_url": results[best_idx].get('link', 'N/A') if results else 'N/A',
                    "mode": "snippet_fallback"
                }
                final_content_for_writer = snippet_text
                anchor_found = True
                break

        # If 'found' was false or all scrapes failed, use pivot_query for next round
        current_query = data.get('pivot_query', f"{title} industry news")

    else: 
        logging.warning(data.get('rationale', "No rationale provided for failure to find anchor."))
 

    if not anchor_found:
        if results and len(results) > 0:
            best_result = results[0] # The top Google result
            
            # We "promote" the snippet to be the 'full_content'
            ledger.anchor_event = {
                "summary": best_result.get('title', 'No Title'),
                "snippet": best_result.get('snippet', 'No Snippet'),
                "full_content": best_result.get('snippet', ''), # PROMOTION
                "source_url": best_result.get('link', 'N/A'),
                "mode": "snippet_fallback"
            }
            return ledger.anchor_event["full_content"]
        else:
            # ULTIMATE FALLBACK: If there are zero search results
            ledger.anchor_event = {
                "summary": "General Technical Summary",
                "full_content": "No external news anchor found. Focus purely on the paper's technical contributions.",
                "mode": "summary_only"
            }
            return ledger.anchor_event["full_content"]
        
  
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

    pivot_res = await client.chat.completions.create(
        model=model_path, 
        messages=[{"role": "user", "content": pivot_prompt}],
        response_format={"type": "json_object"}
    )

    raw_str = pivot_res.choices[0].message.content
    connection_data = extract_rubrics(raw_str)
    logging.info(f"Generated Connection Rubrics: {connection_data}")
    
    # Add these initial constraints to the ledger as the starting state for Phase 3
    ledger.corrective_rubrics.append(f"HOOK CONSTRAINT: {connection_data['hook_rubric']}")
    ledger.corrective_rubrics.append(f"BRIDGE CONSTRAINT: {connection_data['bridge_rubric']}")

    return final_content_for_writer





async def phase_3_drafting(row, ledger, model_path):
    abstract = row.get('Abstract', '')
    introduction = row.get('Introduction', '')
    citation = row.get('citation', '')
    mode = "newsjack" if ledger.anchor_event != "Use general summary mode rubrics (no specific anchor event found)." else "summary"
    
    for attempt in range(5): 
        # Incorporate the "Evolving State" (Connection + Corrective Rubrics)
        active_rubrics = "\n".join([f"- {r}" for r in ledger.corrective_rubrics])
        current_round = attempt + 1

        writer_prompt = f"""
        Persona: Experienced computer science journalist. Audience is expert readers with computer science research backgrounds.

        Core Perspective: Your voice is engaging and narrative. You must prioritize technical significance, expert-level implications, and nuanced breakthroughs. 

        TASK: Write/Refine a 600-800 word news article using these sources: 
        
        SOURCES:
        [Abstract]: {abstract}
        [Introduction]: {introduction}
        [Citation]: {citation}
        [Real-World Anchor Event]: {ledger.anchor_event}

        EVOLVING RUBRICS (CONSTRAINTS TO SATISFY):
        {active_rubrics}

        STRICT EDITORIAL GUIDELINES:
        1. SOURCE HIERARCHY: Every technical claim must be verifiable against the [Abstract], [Introduction], [Citation], and should be the main technical contribution of the paper. 
        2. NO DESCRIPTIVE INFERENCE: Do not add qualitative adjectives (e.g., "groundbreaking," "seamless," "critical") or industry status descriptors (e.g., "the next frontier") unless they are explicitly present in the sources.
        3. NO QUANTITATIVE ASSUMPTIONS: Do not estimate, quantify, or number processes (e.g., "a three-step method" or "several months") unless the specific number or duration is explicitly stated in the sources. 
        4. ANCHOR INTEGRATION:The [Real-World Anchor Event] should be used as an engagement hook, but must not spill over into the technical breakdown of the paper in subsequent paragraphs.
        5. TECHNICAL TERMINOLOGY: Use exact technical terminologies from the paper. Do not use "bridge" terms or analogies that introduce concepts not found in the original text.
        6. STRUCTURE: Follow the Inverted Pyramid. Paragraph 1: Anchor Event. Paragraph 2: Main contributions of paper. Subsequent paragraphs: Technical deep-dive.
        7. Conclude with a motivated Call to Action to access the research paper. Frame the research paper as the main resource for obtaining more information that underpin these findings.
        8. Write like how human journalists write, not like how an academic abstract is structured. Use engaging language but do not sacrifice technical accuracy for the sake of storytelling.

        Format:
        Headline: [Your Headline]
        News Article: [Content]
        End of News Article.
        """
        
        res = await client.chat.completions.create(model=model_path, messages=[{"role": "user", "content": writer_prompt}])
        draft = res.choices[0].message.content

        # The Judge evaluates against the current ledger state
        judge_prompt = f"""
        ROLE: Adversarial Technical Editor.

        [Target Rubrics to Satisfy]: 
        {active_rubrics}

        [Draft]: {draft}
        [Source Abstract]: {abstract}
        [Source Introduction]: {introduction}
        [Source Citation]: {citation}
        [Technical Anchor]: {ledger.technical_anchor}

        TASK: 
        1. Evaluate if the [Draft] satisfies the [Target Rubrics] (Hook, Bridge, and any previous Fixes).
        2. Ensure technical fidelity to the [Source Abstract], [Source Introduction], [Source Citation] and [Technical Anchor].

        EVOLUTION RULE: 
        If the draft fails to meet the rubrics or contains technical errors, mark 'pass': false.
        Generate a new 'Instance-Specific Corrective Rubric' (a direct fix command). Do not suggest examples that does not have a direct attribution in the draft.

        OUTPUT JSON ONLY:
        {{
            "pass": bool,
            "rationale": "Detailed explanation of which rubric or technical nuance was violated.",
            "corrective_rubric": "A specific 'Style' or 'Logic' fix constraint (e.g., 'Change X to Y')."
        }}
        """

        eval_res = await client.chat.completions.create(
            model=JUDGE_MODEL, 
            messages=[{"role": "system", "content": "You are a rigid JSON output engine."},
                      {"role": "user", "content": judge_prompt}], 
            response_format={"type": "json_object"}
        )
        
        raw_content = eval_res.choices[0].message.content

        try:
            feedback = safe_parse_json(raw_content)
            if not feedback:
                # This is your safety net for char 1 errors
                logging.error(f"🚨 Paper {row.get('row_index')}: Judge returned unparseable content. Skipping critique.")
                feedback = {
                    "pass": True, 
                    "rationale": "JSON_PARSE_ERROR: Model returned non-JSON. Self-passing to avoid pipeline stall.",
                    "corrective_rubric": None
                }

            ledger.judge_rationales.append(feedback.get('rationale', 'No rationale.'))
            print(f"👨‍⚖️ Judge Rationale: {feedback.get('rationale')}")

            is_passed = feedback.get('pass', False)
            
            # Record this round's metadata
            round_entry = {
                "mode": mode,
                "round": current_round,
                "action": "stop" if is_passed or current_round == 5 else "regenerate",
                "rationale": feedback.get('rationale', "No rationale provided.")
            }
            ledger.history_log.append(round_entry)
            
            # 3. BRANCHING: YES (Success) -> Output | NO (Critique) -> Evolution
            if is_passed:
                print(f"✅ Draft Passed on Attempt {attempt + 1}!")
                return draft
            
            # Evolution: Add the specific fix constraint to the ledger
            if feedback.get('corrective_rubric'):
                ledger.corrective_rubrics.append(feedback['corrective_rubric'])
                logging.info(f"🔄 Evolution: Added Corrective Rubric - {feedback['corrective_rubric']}")

        except json.JSONDecodeError:
            logging.error(f"Failed to parse Judge JSON.")
            return draft
    
    return draft



async def process_paper(idx, row, model_alias, model_path, output_path):
    async with sem:
        paper_title = row.get('news_title', 'Unknown Title')
        print(f"\n{'='*60}")
        print(f"🚀 STARTING PAPER {idx+1}: {paper_title[:50]}...")
        print(f"{'='*60}")
        
        ledger = RubricLedger()

        try: 
        
            # --- PHASE 1 ---
            print(f"🔍 Phase 1: Analyzing paper and generating technical anchor...")
            init_data = await phase_1_init(row['Abstract'], row['Introduction'], model_path)
            ledger.technical_anchor = init_data['technical_anchor']
            # SAVE THE INITIAL RUBRIC/QUERY
            ledger.search_rubrics.append(init_data['search_query'])
            print(f"✅ Technical Anchor identified: {ledger.technical_anchor}")
            
            # --- PHASE 2 ---
            print(f"🌐 Phase 2: Executing news search for anchor event...")
            retrieved_content = await phase_2_discovery_and_pivot(paper_title, row['Abstract'], row['Introduction'], init_data['search_query'], ledger, model_path)
            # NEW CODE (Dictionary-safe)
            if isinstance(ledger.anchor_event, dict):
                # 1. Force the result to be a string, even if .get() returns None
                # 2. Use 'or' to catch cases where the key exists but is None or ""
                event_text = str(ledger.anchor_event.get('summary') or 'No summary')
                mode = ledger.anchor_event.get('mode', 'unknown')
                
                print(f"🎯 Anchor Event Found ({mode}): {event_text[:100]}...")
            else:
                # Fallback for when anchor_event is a string or something else
                print(f"🎯 Anchor Event: {str(ledger.anchor_event or 'N/A')[:100]}...")

            
            # --- PHASE 3 ---
            print(f"✍️ Phase 3: Commencing Drafting & Self-Refinement Loop...")
            final_article = await phase_3_drafting(row, ledger, model_path)
            
            print(f"✨ COMPLETED PAPER {idx+1}")
            print(f"📊 Final Ledger Corrective Rubrics: {len(ledger.corrective_rubrics)}")
            for i, rub in enumerate(ledger.corrective_rubrics):
                print(f"   └─ Rubric {i+1}: {rub}")
            
            result_payload = {
                "row_index": idx, 
                "generation_model": model_alias,
                "title": paper_title,
                "abstract": row['Abstract'],
                "introduction": row.get('Introduction', ''),
                "citation": row.get('citation', ''),
                "rag_generated_news_article": final_article, 
                "ledger": vars(ledger), 
            }

            # Save results
            await save_result(result_payload, output_path)
            print(f"💾 Progress Saved for Paper {idx+1}")
            
            return result_payload
        
        except Exception as e:
            logging.error(f"❌ CRITICAL ERROR on Paper {idx+1} ({paper_title}): {str(e)}")
            
            # Optional: Save a "Failure" entry so the resume logic knows to skip it next time
            failure_payload = {
                "row_index": idx, 
                "generation_model": model_alias,
                "title": paper_title,
                "abstract": row['Abstract'],
                "introduction": row.get('Introduction', ''),
                "citation": row.get('citation', ''),
                "rag_generated_news_article": "Failed to generate article due to an error.",
                "error": str(e)
            }
            await save_result(failure_payload, output_path)
            return failure_payload



async def save_result(result, file_path):
    """Saves a single result to the JSON file safely using a lock and atomic swap."""
    async with file_lock:
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
        data.append(result)
        
        # Write to a temporary file first, then swap
        temp_path = f"{file_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Atomic rename ensures the file is never "half-written"
        os.replace(temp_path, file_path)


        
async def main(file_path='extracted_papers_summary_5.json', model_file_prefix='rag_final/rag_generated_articles_evolving_rubric_full'):
    os.makedirs('rag_final', exist_ok=True)
    df = pd.read_json(file_path)

    for alias, model_path in EXPERIMENT_MODELS.items():
        # model_file = f"rag_final/rag_generated_articles_evolving_rubric_full_{alias}.json"
        model_file = f"{model_file_prefix}_{alias}.json"

        # RESUME LOGIC (Specific to THIS model file)
        processed_indices = set()
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    data = json.load(f)
                    processed_indices = {item['row_index'] for item in data}
                print(f"🔄 [{alias}] Already finished {len(processed_indices)} papers.")
            except: pass
        else:
            with open(model_file, 'w') as f: json.dump([], f)

        # Only process what's left for this model
        tasks = [process_paper(i, row, alias, model_path, model_file) 
                 for i, row in df.iterrows() if i not in processed_indices]
        
        if tasks:
            print(f"🔬 STARTING: {alias.upper()}")
            # Finish the whole list for GPT before moving to the next loop iteration
            await asyncio.gather(*tasks) 
        else:
            print(f"✅ {alias.upper()} is already done.")

    print("🏁 ALL EXPERIMENTS FINISHED.")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
