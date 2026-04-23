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
    "gpt4o": "openai/gpt-4o",
    "gpt4_1": "openai/gpt-4.1", 
    "llama8b": "meta-llama/llama-3-8b-instruct",
    "llama70b": "meta-llama/llama-3-70b-instruct",
    "mixtral": "mistralai/mistral-small-24b-instruct-2501"
}

# --- STEP 1: SIMPLIFIED PROMPT ---
def create_non_rag_prompt(abstract, introduction, citation):
    """Generates a prompt using ONLY the provided paper content."""
    return f"""
    Persona: You are an experienced computer science journalist writing for an expert technical audience. You focus on technical accuracy, impact framing and audience engagement.
    
    Task: Write a news article (600-900 words) for expert readers with sufficient research backgrounds interested in the Computer Science field.
    
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

def generate_non_rag_article(model_id, abstract, introduction, citation):    
    try:
        prompt = create_non_rag_prompt(abstract, introduction, citation)

        response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def main():
    df = pd.read_json('extracted_papers_summary_5.json').head(50)
    output_path = 'nonrag/nonrag_generated_articles_v5.json'
    all_results = []

    # Nested progress bars: Outer for models, Inner for papers
    model_pbar = tqdm(MODELS.items(), desc="Total Models", unit="model")

    for model_name, model_id in model_pbar:
            model_pbar.set_description(f"Current Model: {model_name}")
            paper_pbar = tqdm(df.iterrows(), total=len(df), desc=f"Progress ({model_name})", leave=False)
            
            for idx, row in paper_pbar:
                generated_article = generate_non_rag_article(
                    model_id, row['Abstract'], row['Introduction'], row['pdf_source']
                )
                
                result_entry = {
                    "model": model_name,
                    "pipeline": "baseline",
                    "paper_index": idx,
                    "abstract": row['Abstract'],
                    "introduction": row['Introduction'],
                    "citation": row['pdf_source'], 
                    "generated_article": generated_article
                }
                
                all_results.append(result_entry)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nAll models completed. Final results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

