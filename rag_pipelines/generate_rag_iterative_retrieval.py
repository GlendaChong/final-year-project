import os
import json
import logging
import asyncio
import httpx
import pandas as pd
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Initialize
load_dotenv()
# Using AsyncOpenAI for non-blocking calls
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_retries=5
)

MODELS = {
    # "gpt4o": "openai/gpt-4o",
    "gpt4_1": "openai/gpt-4.1",
    # "llama8b": "meta-llama/llama-3-8b-instruct",
    # "llama70b": "meta-llama/llama-3-70b-instruct",
    # "mixtral": "mistralai/mistral-small-24b-instruct-2501"
}

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SEARCH_RESULTS_FILE = 'rag_final/rag_one_shot_retrieval.json'

# Limit concurrent API tasks to avoid RateLimitErrors (429)
sem = asyncio.Semaphore(5)


def load_search_bank():
    if os.path.exists(SEARCH_RESULTS_FILE):
        try:
            with open(SEARCH_RESULTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding {SEARCH_RESULTS_FILE}. Starting with empty bank.")
            return {}
    return {}

def save_to_search_bank(bank_data):
    file_dir = os.path.dirname(SEARCH_RESULTS_FILE)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        
    with open(SEARCH_RESULTS_FILE, 'w') as f:
        json.dump(bank_data, f, indent=4)


# --- STAGE 1: BRIDGE ---
async def generate_bridge_queries(model_id, abstract, introduction):
    system_prompt = (
        "You extract research background, challenges, and societal impact information from scientific text. "
        "Focus on the broader problem being addressed, its severity, and why this research matters. "
        "Return strictly the requested JSON schema."
    )
    user_prompt = f"""
    Paper abstract: {abstract}
    Paper introduction: {introduction}

    Task: Identify the real-world problem or challenge this research addresses, and formulate a question that a non-expert (journalist, policymaker, or general public) would ask to understand WHY this research matters.

    Focus on:
    1. What condition/disease/problem affects people in real life?
    2. How widespread or serious is this problem?
    3. What are the human/societal consequences?

    Generate:
    - A claim about the real-world problem (NOT about the research findings)
    - Keywords that a journalist would use (avoid technical jargon)
    - A search query written like a news headline or question someone would ask Google

    Return JSON only:
    {{
      "claim": "[Real-world problem statement with human impact]",
      "keywords": ["[condition in plain language]", "[how many people affected]", "[deaths/costs]", "[why it matters]"],
      "search_query": "[natural language question or news-style query]"
    }}
    """
    response = await openrouter_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# --- STAGE 2: SEARCH ---
async def serper_search_impact(http_client, query, num_results=5):
    url = "https://google.serper.dev/search"
    core_exclusions = "-site:arxiv.org -site:researchgate.net -site:springer.com -site:sciencedirect.com -site:ieee.org -site:nature.com -site:openreviews.net"
    social_exclusions = "-site:linkedin.com -site:twitter.com -site:facebook.com"

    # Formatting the exclusion string for the query
    final_q = f"{query} {core_exclusions} {social_exclusions} "

    payload = {
        "q": final_q,
        "num": 10, 
    }
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        response = await http_client.post(url, headers=headers, json=payload, timeout=10.0)
        results = response.json().get('organic', [])
        return results[:num_results]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []


# --- STAGE 3: SCORING ---
async def score_and_rerank_context(model_id, context_list, keywords, k=3):
    if not context_list:
        return []

    snippets_payload = ""
    for i, item in enumerate(context_list):
        snippet_text = item.get('snippet', 'No snippet available.')
        snippets_payload += f"ID {i}: {snippet_text}\n---\n"

    # Updated prompt for 1-5 Likert Scale
    score_prompt = f"""
    Keywords: {keywords}
    
    You are a Lead Editor. Rate these {len(context_list)} search snippets for inclusion in a news article using a scale of 1 to 5.
    
    SCORING CRITERIA:
    5 - Critical: Essential evidence (hard stats, specific dates, or key quotes).
    4 - High: Strong context (industry trends or specific real-world examples).
    3 - Moderate: Tangentially related (general definitions or vague mentions).
    2 - Low: Minor relevance or overly redundant information.
    1 - Irrelevant: Completely off-topic, broken, or junk text.

    Snippets:
    {snippets_payload}

    Return JSON:
    {{
      "evaluations": [
        {{"id": 0, "score": 4, "reason": "Short explanation of why"}}
      ]
    }}
    """

    try:
        res = await openrouter_client.chat.completions.create(
            model=model_id, 
            messages=[{"role": "user", "content": score_prompt}],
            response_format={"type": "json_object"}
        )

        result_data = json.loads(res.choices[0].message.content)
        evals = {item['id']: item for item in result_data.get('evaluations', [])}

        scored_list = []
        for i, item in enumerate(context_list):
            eval_item = evals.get(i, {"score": 0})
            item['score'] = eval_item.get('score', 0)
            item['filter_reason'] = eval_item.get('reason', 'N/A')
            scored_list.append(item)

        # ARRANGE: Sort by score descending
        scored_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Return only the top k
        return scored_list[:k]

    except Exception as e:
        logging.error(f"Batch scoring error: {e}")
        return context_list[:k] # Return raw top 3 if scoring fails
    

# --- STAGE 4: GENERATION ---
async def generate_final_article(model_id, abstract, introduction, citation, external_context):
    prompt = f"""
    Persona: Experienced computer science journalist. Audience is expert readers with computer science research backgrounds.

    TASK: Write/Refine a 600-900 word news article using these sources: 
    
    SOURCES:
    [Abstract]: {abstract}
    [Introduction]: {introduction}
    [Citation]: {citation}

    [Filtered Real-World Context]
    {external_context}

    STRICT EDITORIAL GUIDELINES:
    1. SOURCE HIERARCHY: Every technical claim must be verifiable against the [Abstract], [Introduction], [Citation] and [Filtered Real-World Context]
    2. NO DESCRIPTIVE INFERENCE: Do not add qualitative adjectives (e.g., "groundbreaking," "seamless," "critical") or industry status descriptors (e.g., "the next frontier") unless they are explicitly present in the sources.
    3. NO QUANTITATIVE ASSUMPTIONS: Do not estimate, quantify, or number processes (e.g., "a three-step method" or "several months") unless the specific number or duration is explicitly stated in Source A or B. 
    5. TECHNICAL TERMINOLOGY: Use the exact terminology from the paper. Do not use "bridge" terms or analogies that introduce concepts not found in the original text.
    6. STRUCTURE: Follow the Inverted Pyramid. Paragraph 1: Real-World Context. Paragraph 2: Main contributions of paper. Subsequent paragraphs: Technical deep-dive.
    7. Conclude with a motivated Call to Action to access the research paper. Frame the research paper as the main resource for obtaining more information that underpin these findings.

    
    Format:
    Headline: [Your Headline]
    News Article: [Content]
    End of News Article.
    """

    try:    
        response = await openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Generation failed for {model_id}: {e}")
        return f"ERROR: Generation failed for {model_id}"   

        
# --- UPDATED WORKER UNIT ---
async def process_paper(idx, row, model_key, model_id, http_client):
    async with sem: 
        title = row.get('news_title', 'Unknown Title')
        abstract, intro, citation = row.get('Abstract', ''), row.get('Introduction', ''), row.get('pdf_source', '')
        
        logging.info(f"🚀 [{model_key}] Processing Paper {idx}: {title}")
        
        # 1. Initial Bridge
        bridge = await generate_bridge_queries(model_id, abstract, intro)
        keywords = bridge.get('keywords', 'N/A')
        
        # --- CONTINUOUS RETRIEVAL LOOP ---
        top_k_context = []
        retry_count = 0
        
        # Define different search "intents" to broaden/sharpen the search
        search_intents = [
            bridge.get('search_query', 'N/A'), # Intent 0: Original query
            f"recent news coverage and major events related to {bridge.get('keywords', [''])[0]} 2024..2026", # Intent 1: Anchor Events
            f"real-world implementation challenges and case studies of {bridge.get('claim', 'N/A')}" # Intent 2: Implementation Challenges
        ]

        while len(top_k_context) < 3 and retry_count < len(search_intents):
            current_query = search_intents[retry_count]
            logging.info(f"🔍 Retry {retry_count} for Paper {idx} using query: {current_query}")
            
            raw_results = await serper_search_impact(http_client, current_query)
            scored_results = await score_and_rerank_context(model_id, raw_results, keywords)
            
            # Only accept "Strong" or "Critical" context (Score >= 7)
            new_valid_snippets = [s for s in scored_results if s.get('score', 0) >= 4]
            
            # Deduplicate by link and add to our collection
            existing_links = {s['link'] for s in top_k_context}
            for s in new_valid_snippets:
                if s['link'] not in existing_links:
                    top_k_context.append(s)
            
            retry_count += 1
            if len(top_k_context) >= 3: break

        # Final selection: Best 3 high-quality snippets
        top_k_context = sorted(top_k_context, key=lambda x: x.get('score', 0), reverse=True)[:3]

        # 2. Generation (One-Shot with the perfected snippets)
        final_article = await generate_final_article(model_id, abstract, intro, citation, top_k_context)


        return {
            "row_index": idx,
            "generation_model": model_key,
            "pipeline_model_id": model_id,
            "original_title": title,
            "context_used_count": len(top_k_context),
            "snippets_used": top_k_context,
            "abstract": abstract,
            "introduction": intro,
            "rag_generated_news_article": final_article,
            "citation": citation,
            "retries_performed": retry_count - 1
        }


# --- MAIN RUNNER ---
async def main():
    df = pd.read_json('extracted_papers_summary_5.json').head(30)
    trace_results = []

    async with httpx.AsyncClient() as http_client:
        for model_key, model_id in MODELS.items():
            logging.info(f"=== STARTING ASYNC PIPELINE FOR MODEL: {model_key} ===")
            
            tasks = [process_paper(idx, row, model_key, model_id, http_client) for idx, row in df.iterrows()]
            
            # Execute concurrently
            model_results = await asyncio.gather(*tasks)
            trace_results.extend([r for r in model_results if r])

            # Identifies cases where no snippets scored above 2
            failures = [r for r in model_results if all(s.get('score', 0) <= 2 for s in r['snippets_used'])]
            if failures:
                os.makedirs('rag_final', exist_ok=True)
                with open(f'rag_final/failure_autopsy_iterative_retrieval_{model_key}.json', 'w') as f:
                    json.dump(failures, f, indent=4)
                    logging.info(f"Saved {len(failures)} failure cases to autopsy file.")

            trace_results.extend([r for r in model_results if r])

            # Intermediate save
            os.makedirs('rag_final', exist_ok=True)
            with open('rag_final/rag_generated_articles_iterative_retrieval.json', 'w') as f:
                json.dump(trace_results, f, indent=4)
                
        pd.DataFrame(trace_results).to_csv('rag_final/rag_generated_articles_iterative_retrieval.csv', index=False)
        print("Done! Results saved in rag_final/rag_generated_articles_iterative_retrieval.csv and rag_final/rag_generated_articles_iterative_retrieval.json.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
