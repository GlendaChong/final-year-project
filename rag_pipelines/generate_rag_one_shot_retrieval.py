import asyncio
import os
import re
import httpx
import json
import logging
import requests
import pandas as pd
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Initialize
load_dotenv()
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
    # Dynamically get the directory from the file path constant
    file_dir = os.path.dirname(SEARCH_RESULTS_FILE)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        
    with open(SEARCH_RESULTS_FILE, 'w') as f:
        json.dump(bank_data, f, indent=4)



async def generate_bridge_queries(model_id, abstract, introduction):
    """
    Extracts human-centric claims and journalistic search queries 
    from a scientific paper to support news article generation.
    """
    
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
    

async def serper_search_impact(query, num_results=5):
    url = "https://google.serper.dev/search"
    core_exclusions = "-site:arxiv.org -site:researchgate.net -site:springer.com -site:sciencedirect.com -site:ieee.org -site:nature.com -site:openreviews.net"
    social_exclusions = "-site:linkedin.com -site:twitter.com -site:facebook.com"

    # Formatting the exclusion string for the query
    final_q = f"{query} {core_exclusions} {social_exclusions}"

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

        # Sort by score descending
        scored_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Return only the top k
        return scored_list[:k]

    except Exception as e:
        logging.error(f"Batch scoring error: {e}")
        return context_list[:k] # Return raw top 3 if scoring fails
    



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

    


async def process_paper(semaphore, model_id, model_key, row, idx, search_bank, http_client):
    """Processes a single paper end-to-end within a concurrency limit."""
    async with semaphore:
        paper_id = f"paper_{idx}"
        title = row.get('news_title', 'Unknown Title')
        abstract = row.get('Abstract', '')
        introduction = row.get('Introduction', '')
        citation = row.get('pdf_source', '')

        logging.info(f"[{model_key}] STARTING Paper {idx}: {title}")

        # STEP 1: Bridge
        bridge_data = await generate_bridge_queries(model_id, abstract, introduction)
        claim = bridge_data.get('claim', 'N/A')
        keywords = bridge_data.get('keywords', 'N/A')
        queries = bridge_data.get('search_query', 'N/A')

        # STEP 2: Search
        if paper_id in search_bank:
            all_raw_context = search_bank[paper_id]
        else:
            all_raw_context = []
            # Ensure queries is a list to avoid iterating over characters of a string
            query_list = queries if isinstance(queries, list) else [queries]
            for q in query_list:
                all_raw_context.extend(await serper_search_impact(q))
        
        # STEP 3: Scoring
        top_k_context = await score_and_rerank_context(model_id, all_raw_context, keywords)

        # Fallback if empty
        if not top_k_context:
            logging.info(f"No context found for {idx}: {claim}. Running broad fallback search...")
            fallback_query = f"Latest news and real world impact of {claim}"
            fallback_raw = await serper_search_impact(fallback_query)
            top_k_context = await score_and_rerank_context(model_id, fallback_raw, claim)

        # STEP 4: Generation
        final_article = await generate_final_article(
            model_id, abstract, introduction, citation, top_k_context
        )

        logging.info(f"[{model_key}] FINISHED Paper {idx}")
        
        return {
            "row_index": idx,
            "generation_model": model_key,
            "pipeline_model_id": model_id,
            "original_title": title,
            "bridge_queries": queries,
            "claim": claim,
            "keywords": keywords,
            "retrieved_snippets": all_raw_context,
            "snippets_used": top_k_context,
            "context_used_count": len(top_k_context),
            "abstract": abstract,
            "introduction": introduction,
            "rag_generated_news_article": final_article
        }


async def main():
    df = pd.read_json('extracted_papers_summary_5.json') 
    search_bank = load_search_bank()
    
    semaphore = asyncio.Semaphore(5)

    async with httpx.AsyncClient() as http_client:
        for model_key, model_id in MODELS.items():
            logging.info(f"=== LAUNCHING ASYNC BATCH FOR MODEL: {model_key} ===")
            
            tasks = []
            for idx, row in df.iloc[:30].iterrows():
                tasks.append(
                    process_paper(semaphore, model_id, model_key, row, idx, search_bank, http_client)
                )

            results = await asyncio.gather(*tasks)

            for res in results:
                paper_id = f"paper_{res['row_index']}"
                if paper_id not in search_bank:
                    search_bank[paper_id] = res['retrieved_snippets']

            save_to_search_bank(search_bank)
            
            # Save the batch results
            with open('rag_final/rag_generated_articles_one_shot_retrieval.json', 'w') as f:
                json.dump(results, f, indent=4)
                
            pd.DataFrame(results).to_csv('rag_final/rag_generated_articles_one_shot_retrieval.csv', index=False)
            
            logging.info(f"Full batch for {model_key} completed and saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())