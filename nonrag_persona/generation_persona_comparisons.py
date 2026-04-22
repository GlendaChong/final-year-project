import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Initialize
load_dotenv()
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"), 
)
# 1. Define your model suite
MODELS = {
    # "gpt4o": "openai/gpt-4o",
    "gpt4_1": "openai/gpt-4.1", 
    # "llama8b": "meta-llama/llama-3-8b-instruct",
    "llama70b": "meta-llama/llama-3-70b-instruct",
    # "mixtral": "mistralai/mistral-small-24b-instruct-2501"
}

# --- TEST CASE DEFINITIONS ---
# Test Case 1 is NIL/NIL (Baseline), 2 and 3 add the persona/audience strings
TEST_CASES = [
    {"id": 1, "persona": "NIL", "audience": "NIL"},
    {"id": 2, "persona": "Computer Science Journalist", "audience": "NIL"},
    {"id": 3, "persona": "Computer Science Journalist", "audience": "Expert Computer Science Audience"}, 
    {"id": 4, "persona": "Academic Peer Reviewer", "audience": "Computer Science Audience"}, 
    {"id": 5, "persona": "Computer Science Technical Lead", "audience": "Computer Science Audience"},
    {"id": 6, "persona": "Educator", "audience": "Computer Science Audience"},
]

# --- REFINED BEHAVIORAL ARCHETYPES ---
PERSONA_ARCHETYPES = {
    2: {
        "voice": "Informative, objective, and neutral.",
        "focus": "General accessibility, high-level impact, and balanced reporting.",
        "instruction": "Follow the inverted pyramid structure. Prioritize the 'Who, What, Where, When, Why' without assuming deep technical expertise."
    },
    3: {
        "voice": "Sophisticated, analytical, and professional.",
        "focus": "Technical significance, expert-level implications, and nuanced breakthroughs.",
        "instruction": "Maintain a high floor of technical jargon. Focus on why this matters to the broader CS research community while keeping a journalistic flow."
    },
    4: {
        "voice": "Critical, analytical, and objective.",
        "focus": "Methodological rigor, benchmarking data, and scientific limitations.",
        "instruction": "Avoid hype; use formal academic language. Frame the breakthrough within the context of existing literature."
    },
    5: {
        "voice": "Practical, decisive, and efficiency-oriented.",
        "focus": "Implementation bottlenecks, scalability, and system integration.",
        "instruction": "Focus on the 'how-to' and the engineering cost-benefit. Use industry-standard terminology for production environments."
    },
    6: {
        "voice": "Encouraging, clear, and explanatory.",
        "focus": "Conceptual intuition, foundational principles, and analogies.",
        "instruction": "Explain complex jargon using relatable real-world metaphors. Prioritize high-level understanding over dense concepts."
    }
}

RUN_SIZE = 50

# --- STEP 1: SIMPLIFIED PROMPT ---
def create_non_rag_prompt(abstract, introduction, citation, persona, audience, tc_id):
    """Generates a prompt incorporating Persona and Target Audience."""
    
    persona_line = f"You are a {persona} writing for {audience}."
    
    # Get the specific behavioral push for this persona
    archetype = PERSONA_ARCHETYPES.get(tc_id, {
        "voice": "Professional and journalistic.",
        "focus": "General impact and broad technical accuracy.",
        "instruction": "Maintain a balanced journalistic perspective."
    })
    
    return f"""
    {persona_line}
    Core Perspective: Your voice is {archetype['voice']} You must prioritize {archetype['focus']}
        
    Task: Write a news article (600-900 words) for readers with sufficient research backgrounds interested in the Computer Science field.
    
    [Source Content: Paper Abstract]
    {abstract}
    
    [Source Content: Paper Introduction]
    {introduction}
    
    [Main Resesarch Paper Citation]: {citation}
    
    Guidelines:
    - Style: Professional, informative. Inverted pyramid structure, starting with the main contributions and impact, then diving into the details. 
    - Citation: Conclude with a motivated Call to Action to access the research paper. Frame the research paper as the main resource for obtaining more information that underpin these findings.
    
    Format:
    Headline: [Your Headline]
    News Article: [Content]
    End of News Article.
    """

def generate_non_rag_article(model_id, abstract, introduction, citation, persona, audience, tc_id):    
    try:
        prompt = create_non_rag_prompt(abstract, introduction, citation, persona, audience, tc_id)

        response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def main():
    df = pd.read_json('extracted_papers_summary_5.json')
    if RUN_SIZE:
        df = df.head(RUN_SIZE)

    output_path = 'nonrag_persona/diff_persona_generated_articles_t2_vs_t3.json'
    all_results = []

    # Nested progress bars: Outer for models, Inner for papers
    model_pbar = tqdm(MODELS.items(), desc="Total Models", unit="model")

    for model_name, model_id in model_pbar:
            model_pbar.set_description(f"Current Model: {model_name}")
            
            for config in TEST_CASES:
                tc_id = config["id"]
                persona = config["persona"]
                audience = config["audience"]

                paper_pbar = tqdm(
                    df.iterrows(), 
                    total=len(df), 
                    desc=f"TC{tc_id} ({model_name})", 
                    leave=False
                )
                
                for idx, row in paper_pbar:
                    generated_article = generate_non_rag_article(
                        model_id, row['Abstract'], row['Introduction'], row['pdf_source'], persona, audience, tc_id
                    )
                    
                    result_entry = {
                        "model": model_name,
                        "pipeline": "baseline",
                        "paper_index": idx,
                        "persona": persona,
                        "audience": audience,
                        "abstract": row['Abstract'],
                        "introduction": row['Introduction'],
                        "citation": row['pdf_source'], 
                        "generated_article": generated_article, 
                        "persona_version": f"t{tc_id}"

                    }
                    
                    all_results.append(result_entry)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nAll models completed. Final results saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

