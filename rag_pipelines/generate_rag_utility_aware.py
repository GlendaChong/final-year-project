import os
import json
import logging
import asyncio
import httpx
import pandas as pd
import re
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
# JUDGE_MODEL = "anthropic/claude-3.5-sonnet"
JUDGE_MODEL = "deepseek/deepseek-r1-0528"


sem = asyncio.Semaphore(5)


def extract_json(text):
    try:
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(text)
    except:
        return None

# STAGE 1: BRIDGE
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
    async with sem:
        response = await openrouter_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
    return json.loads(response.choices[0].message.content)


# STAGE 2: SEARCH
async def serper_search_impact(query, num_results=5):
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
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(url, headers=headers, json=payload, timeout=10.0)
            results = response.json().get('organic', [])
            return results[:num_results]
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

# STAGE 3: SCORING
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
    


# STAGE 4: GENERATION
async def generate_final_article(model_id, abstract, introduction, citation, external_context):
    prompt = f"""
    Persona: Experienced computer science journalist. Audience is expert readers with computer science research backgrounds.

    Core Perspective: Your voice is sophisticated, analytical, and professional. You must prioritize technical significance, expert-level implications, and nuanced breakthroughs.

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
    async with sem:
        try:    
            response = await openrouter_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return f"ERROR: Generation failed"
    

async def run_utility_judge(model_id, article, retrieved_content, prompt_template):
    """
    PHASE 3: The Judge. Uses your existing 4b_rag_utility logic 
    to decide if the draft is good enough.
    """
    # Format using your existing template style
    full_prompt = prompt_template.format(
        generated_content=article, 
        anchor_event=retrieved_content
    )
    
    async with sem:
        response = await openrouter_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": full_prompt + "\n\nReturn ONLY JSON."}],
            response_format={"type": "json_object"}, # Best for structured output 
            temperature=0.0 # Deterministic output for evaluation
        )

        data = extract_json(response.choices[0].message.content)
        
        # Ensure we return a dictionary with lowercase keys
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items()}
        
        return {"score": "N/A", "rationale": "Parsing failed"}
        


async def generate_pivot_query(model_id, rationale, abstract, introduction, previous_queries):
    """
    Refined pivot query generator that avoids previous failures.
    previous_queries: a list of strings already used.
    """
    prompt = f"""
    [TECHNICAL PAPER ABSTRACT]: {abstract}
    [TECHNICAL PAPER INTRODUCTION]: {introduction}
    [CRITIQUE FROM EDITOR]: {rationale}
    
    [PREVIOUS SEARCHES (FAILED)]: {previous_queries}
    
    TASK:
    The previous searches failed to bridge the gap between the technical paper's content and the real-world impact.
    Identify the 'Missing Link' (e.g., a specific industry pain point, a statistical gap, or a recent regulatory change).
    
    Generate a highly specific 'Search String' that avoids the keywords used in the failed queries. 
    Focus on finding:
    - Real-world "case studies"
    - "Economic impact" data
    - "Implementation challenges" in specific industries
    
    Return JSON: 
    {{
      "missing_link_analysis": "Briefly explain what information is missing",
      "pivot_query": "The new optimized search string"
    }}
    """
    async with sem:
        res = await openrouter_client.chat.completions.create(
            model=model_id, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return extract_json(res.choices[0].message.content)


async def process_paper(idx, row, model_key, model_id):
    title = row.get('news_title', 'Unknown Title')
    abstract, intro, citation = row.get('Abstract', ''), row.get('Introduction', ''), row.get('pdf_source', '')
    
    logging.info(f"🚀 [{model_key}] Processing Paper {idx}: {title}")
    
    bridge = await generate_bridge_queries(model_id, abstract, intro)
    keywords = bridge.get('keywords', 'N/A')
    
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
        
        raw_results = await serper_search_impact(current_query)
        scored_results = await score_and_rerank_context(model_id, raw_results, keywords)
        
        # Only accept "Strong" or "Critical" context (Score >= 4)
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

    # Drafting loop
    article = await generate_final_article(model_id, abstract, row.get('Introduction', ''), row.get('pdf_source', ''), top_k_context)
    
    # Utility Judgment (The Judge)
    retrieved_text = "\n\n".join([
        f"Source Title: {c.get('title', 'N/A')}\n"
        f"URL: {c.get('link', 'N/A')}\n"
        f"Snippet: {c.get('snippet', 'N/A')}" 
        for c in top_k_context
    ])

    with open(f"prompts/rag_with_evolving_rubrics/4b_rag_utility.txt", 'r') as f:
        utility_prompt_template = f.read()
    judgment = await run_utility_judge(model_id, article, retrieved_text, utility_prompt_template)

    final_article = article # Default to original
    was_refined = False
    final_score = judgment.get('score', 'N/A')
    rationale = judgment.get('rationale', 'N/A')
    
    # 3. Success Check (e.g., Score < 4 triggers evolution)
    logging.info(f"⚖️ Utility Judgment for Paper {idx}: Score = {final_score}, Rationale = {rationale}")

    if judgment.get('score', 5) <= 4:
        logging.info(f"🔄 Phase 3 Failure: Pivoting for better context.")
        
        # 1. PIVOT: Search for the missing link
        pivot = await generate_pivot_query(model_id, judgment.get('rationale', ''), abstract, intro, search_intents[:retry_count])
        
        new_raw = await serper_search_impact(pivot.get('pivot_query', ''))
        logging.info(f"🎯 PIVOTING: New targeted query -> {new_raw}")
        
        # 2. RERANK: Score the new results
        new_context = await score_and_rerank_context(model_id, new_raw, keywords)
        
        # 3. CONSOLIDATE & RE-SORT: Put all found snippets together
        all_potential_context = top_k_context + new_context
        
        # Deduplicate by URL to be safe
        unique_context = {c['link']: c for c in all_potential_context}.values()
        
        # FINAL RERANK: Pick the best 3-5 from the entire pool
        combined_context = sorted(unique_context, key=lambda x: x.get('score', 0), reverse=True)[:4]

        # 4. EVOLUTION: Generate the refined article
        final_article = await generate_final_article(model_id, abstract, intro, citation, combined_context)
        was_refined = True
        
        # 5. FINAL VERDICT: Let the judge see the final product
        new_retrieved_text = "\n\n".join([
            f"Source: {c.get('title')}\nSnippet: {c.get('snippet')}" 
            for c in combined_context
        ])
        re_judgment = await run_utility_judge(model_id, final_article, new_retrieved_text, utility_prompt_template)


    final_article = article if not was_refined else final_article

    return {
        "row_index": idx,
        "generation_model": model_key,
        "pipeline_model_id": model_id,
        "original_title": title,
        "abstract": abstract,
        "introduction": intro,
        "citation": citation,
        
        # Retrieval Metadata
        "retries_performed": retry_count - 1,
        "context_used_count": len(combined_context) if was_refined else len(top_k_context),
        "snippets_used": combined_context if was_refined else top_k_context,
        
        # Articles (Keep both for delta analysis)
        "previous_generated_article": article,
        "rag_generated_news_article": final_article,
        
        # Phase 3: Utility Evaluation Data
        "was_refined": was_refined,
        "initial_utility_score": judgment.get('score', 'N/A'),
        "initial_utility_rationale": judgment.get('rationale', 'N/A'),
        "final_utility_score": re_judgment.get('score', 'N/A') if was_refined else judgment.get('score', 'N/A'),
        "final_utility_rationale": re_judgment.get('rationale', 'N/A') if was_refined else judgment.get('rationale', 'N/A')
    }



async def main():
    df = pd.read_json('extracted_papers_summary_5.json')
    trace_results = []
    
    for model_key, model_id in MODELS.items():
        logging.info(f"=== STARTING ASYNC PIPELINE FOR MODEL: {model_key} ===")
        
        # Create tasks for all 10 papers for the current model
        tasks = [process_paper(idx, row, model_key, model_id) for idx, row in df.iterrows()]
        
        # Execute concurrently
        model_results = await asyncio.gather(*tasks)
        trace_results.extend([r for r in model_results if r])

        os.makedirs('rag_final', exist_ok=True)
        with open('rag_final/rag_generated_articles_utility_full.json', 'w') as f:
            json.dump(trace_results, f, indent=4)
            
    pd.DataFrame(trace_results).to_csv('rag_final/rag_generated_articles_utility_full.csv', index=False)
    print("Done! Results saved in rag_final/rag_generated_articles_utility_full.csv and rag_final/rag_generated_articles_utility_full.json.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())