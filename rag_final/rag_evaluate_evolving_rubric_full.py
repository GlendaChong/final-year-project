import os, json, logging, asyncio, pandas as pd, re, time
from tqdm.asyncio import tqdm  # Specialized async progress bar
from pydantic import BaseModel, ConfigDict, model_validator
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- SETTINGS ---
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# JUDGE_MODEL = "anthropic/claude-3.5-sonnet"
JUDGE_MODEL = "deepseek/deepseek-r1"

CONCURRENT_PAPERS = 5 # How many papers to evaluate at once
METRIC_KEYS = ['1a_accuracy', '1b_technical_distortion', '2a_novelty_emphasis', 
               '2b_scientific_significance', '3a_engagement_hook_strength', 
               '3b_logical_attractiveness', '3c_call_to_action', '4a_rag_relevance', '4b_rag_utility']

MODEL_ALIASES = ["gpt4_1", "gpt4o", "gemini_pro", "llama_70b", "deepseek_v3", "deepseek_r1"]
sem = asyncio.Semaphore(CONCURRENT_PAPERS)

class EvalResult(BaseModel):
    rationale: str
    score: int
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def lowercase_keys(cls, data):
        if isinstance(data, dict): return {k.lower(): v for k, v in data.items()}
        return data


def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


async def run_judge(prompt_template, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            full_prompt = prompt_template.format(**kwargs)
            
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": full_prompt + "\n\nReturn ONLY JSON."}],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=30.0 
            )
            
            raw_content = response.choices[0].message.content
            cleaned_json = extract_json(raw_content)
            return EvalResult.model_validate_json(cleaned_json)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2 
                logging.warning(f"⚠️ Retry {attempt+1} after error: {e}")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"❌ Final failure for metric: {str(e)}")
                return EvalResult(rationale=f"Failed after {max_retries} attempts: {str(e)}", score=0)
            


async def evaluate_entry(prompts, row, pbar, model_alias):
    async with sem:
        paper_id = row.get('row_index')
        
        pbar.set_postfix({"paper": paper_id, "model": model_alias})
        
        article = row.get('rag_generated_news_article', '')
        ledger = row.get('ledger', {})
        anchor_event = ledger.get('anchor_event', 'N/A')

        tasks = [
            run_judge(prompts[k], 
                      abstract=row.get('abstract',''), 
                      intro=row.get('introduction',''), 
                      generated_content=article, 
                      pdf_source=row.get('citation',''),
                      anchor_event=str(anchor_event)) 
            for k in METRIC_KEYS
        ]
        
        results = await asyncio.gather(*tasks)
        
        eval_data = {"paper_index": paper_id, "model_alias": model_alias}
        for i, key in enumerate(METRIC_KEYS):
            eval_data[f"{key}_score"] = results[i].score
            eval_data[f"{key}_rationale"] = results[i].rationale
        
        pbar.update(1)
        return eval_data
    

async def main():
    os.makedirs('rag_final/evaluations', exist_ok=True)
    
    # Load prompts
    loaded_prompts = {}
    for key in METRIC_KEYS:
        with open(f"prompts/rag_with_evolving_rubrics/{key}.txt", 'r') as f:
            loaded_prompts[key] = f.read()

    # Loop through each model's result file
    for alias in MODEL_ALIASES:
        input_file = f"rag_final/rag_generated_articles_evolving_rubric_full_{alias}.json"
        output_base = f"rag_final/evaluations/eval_{alias}"
        
        if not os.path.exists(input_file):
            print(f"⏭️ Skipping {alias}: File not found.")
            continue

        with open(input_file, 'r') as f:
            data = json.load(f)

        print(f"\n🚀 Evaluating {alias.upper()} ({len(data)} papers) using {JUDGE_MODEL}...")

        # Use TQDM for visual progress and ETA
        with tqdm(total=len(data), desc=f"📊 {alias}") as pbar:
            eval_tasks = [evaluate_entry(loaded_prompts, row, pbar, alias) for row in data]
            final_results = await asyncio.gather(*eval_tasks)

        # Save results
        df = pd.DataFrame(final_results)
        df.to_csv(f"{output_base}.csv", index=False)
        with open(f"{output_base}.json", 'w') as f:
            json.dump(final_results, f, indent=4)
        
        print(f"✅ Saved results for {alias}")


if __name__ == "__main__":
    asyncio.run(main())