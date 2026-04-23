import logging
import requests

# Extract major claim from scientific paper text
def extract_major_claim(text, client): 
    prompt_to_retrieval = (
        """
            You extract concise, search-ready claims and keywords from scientific text.
            Return strictly the requested JSON schema.
            User Template
            Paper title: {title}
            Paper (excerpt):
            {paper}

            Task: Identify one high-level, real-world-relevant claim (1 sentence, <=30 words),
            and 5–8 search keywords (single words or short phrases), and a concise search query (<=12 words)
            to find timely context (news, policy, industry impact).

            Return JSON only:
            {
                "claim": "...",
                "keywords": ["k1", "k2", ...],
                "search_query": "..."
            }

            Expected JSON Schema
            claim:
            1 sentence, ≤30 words
            No citations or URLs

            keywords:
            5–8 items
            Short phrases only

            search_query:
            ≤12 words
            Optimized for web search
        """
    )

    response = client.responses.create(
        model="gpt-4o",
        input=prompt_to_retrieval,
        temperature=0.0,  # deterministic output
    )

    logging.info(f"Major claim extraction response: {response.output_text}")
    return response.output_text


# Retrieve relevant context from web using major claim
def serper_search(claim):
    api_key = os.getenv('SERPER_API_KEY') 
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    query = f"{claim}. surveys statistics expert opinions societal impact. not research papers of the claim itself."
    payload = {
        "q": query
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  
    
    data = response.json()
    results = data.get('organic', []) # 'organic' key usually contains the main search results list
    print(results)
    return results
