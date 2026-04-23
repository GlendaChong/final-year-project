import os
import asyncio
import json
import re
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from openai import AsyncOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.asyncio import tqdm

load_dotenv()
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"), 
)

# CONFIGURATION
INPUT_PATH = 'nonrag/nonrag_generated_articles_no_persona_baseline.json'
OUTPUT_PATH = 'nonrag/nonrag_evaluation_no_persona_baseline'
JUDGE_MODEL = "deepseek/deepseek-r1"
MAX_CONCURRENT_TASKS = 10 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

class EvalResult(BaseModel):
    rationale: str
    score: int
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def lowercase_keys(cls, data):
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items()}
        return data

def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    except Exception:
        return text


async def run_judge(metric_name, paper_idx, prompt_template, **kwargs):
    async with semaphore: 
        full_prompt = prompt_template.format(**kwargs)
        try:
            response = await asyncio.wait_for(
                openrouter_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": full_prompt + "\n\nReturn ONLY JSON."}],
                    response_format={"type": "json_object"}
                ),
                timeout=120.0 
            )
            
            raw_content = response.choices[0].message.content
            raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
            cleaned_json = extract_json(raw_content)
            print(f"  [Success] Paper {paper_idx}: {metric_name} evaluated. Extracted JSON: {cleaned_json[:50]}...") # Show a snippet of the JSON for debugging
            result = EvalResult.model_validate_json(cleaned_json)
            
            return result

        except asyncio.TimeoutError:
            print(f"  [TIMEOUT ERROR] Paper {paper_idx}: {metric_name} timed out.")
            return EvalResult(rationale="Request timed out after 90s", score=0)
        except Exception as e:
            print(f"  [LLM ERROR] Paper {paper_idx}: {metric_name} failed: {e}")
            return EvalResult(rationale=f"Error: {str(e)}", score=0)

# Parallel Execution for Each Entry
async def evaluate_entry(prompts, row):
    paper_id = row.get('paper_index')
    gen_model = row.get('model')
    persona = row.get('persona_version', 'N/A')
    
    tasks = [
        run_judge('1a_accuracy', paper_id, prompts['1a_accuracy'], abstract=row['abstract'], intro=row['introduction'], pdf_source=row['citation'], generated_content=row['generated_article']),
        run_judge('1b_tech_distortion', paper_id, prompts['1b_technical_distortion'], abstract=row['abstract'], intro=row['introduction'], generated_content=row['generated_article']),
        run_judge('2a_novelty', paper_id, prompts['2a_novelty_emphasis'], abstract=row['abstract'], intro=row['introduction'], pdf_source=row['citation'], generated_content=row['generated_article']), 
        run_judge('2b_scientific_sig', paper_id, prompts['2b_scientific_significance'], abstract=row['abstract'], intro=row['introduction'], pdf_source=row['citation'], generated_content=row['generated_article']),
        run_judge('3a_hook', paper_id, prompts['3a_engagement_hook_strength'], generated_content=row['generated_article']),
        run_judge('3b_logic', paper_id, prompts['3b_logical_attractiveness'], generated_content=row['generated_article']),
        run_judge('3c_cta', paper_id, prompts['3c_call_to_action'], pdf_source=row['citation'], generated_content=row['generated_article']),
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "paper_index": paper_id,
        "generation_model": gen_model,
        "persona_version": persona,
        "1a_accuracy_score": results[0].score,
        "1b_technical_distortion_score": results[1].score,
        "2a_novelty_emphasis_score": results[2].score,
        "2b_scientific_significance_score": results[3].score,
        "3a_engagement_hook_strength_score": results[4].score,
        "3b_logical_attractiveness_score": results[5].score,
        "3c_call_to_action_score": results[6].score,
        "full_results": [r.rationale for r in results]
    }


async def main():
    metric_keys = ['1a_accuracy', '1b_technical_distortion', '2a_novelty_emphasis', '2b_scientific_significance', '3a_engagement_hook_strength', '3b_logical_attractiveness', '3c_call_to_action']
    loaded_prompts = {}
    
    for key in metric_keys:
        path = f"prompts/non_rag/{key}.txt"
        if os.path.exists(path):
            with open(path, 'r') as f:
                loaded_prompts[key] = f.read()
        else:
            print(f"Warning: Prompt file {path} missing!")

    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)

    print(f"Starting evaluation of {len(data)} entries using {JUDGE_MODEL}...")
    
    eval_tasks = [evaluate_entry(loaded_prompts, row) for row in data]
    
    final_results = []
    
    for f in tqdm(asyncio.as_completed(eval_tasks), total=len(eval_tasks), desc="Processing Papers"):
        res = await f
        final_results.append(res)
    
    df = pd.DataFrame(final_results)
    df.to_csv(OUTPUT_PATH + '.csv', index=False)
    
    # Analysis
    print("\n--- RQ Analysis: Mean Quality Scores ---")
    score_cols = [col for col in df.columns if col.endswith('_score')]
    df['mean_quality'] = df[score_cols].mean(axis=1)
    
    pivot = df.pivot_table(index='persona_version', columns='generation_model', values='mean_quality')
    print(pivot)
    
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Persona vs. Model Quality (Mean Score)")
    plt.savefig(OUTPUT_PATH + '_heatmap.png')
    print(f"\nHeatmap saved to {OUTPUT_PATH}_heatmap.png")

if __name__ == "__main__":
    asyncio.run(main())