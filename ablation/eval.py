import os, json, logging, asyncio, pandas as pd, re, time
from tqdm.asyncio import tqdm
from pydantic import BaseModel, ConfigDict, model_validator
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- SETTINGS ---
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("OPENROUTER_API_KEY")
)

JUDGE_MODELS = {
    # "deepseek_chat": "deepseek/deepseek-chat",      # Baseline
    # "gemini_2.5_flash": "google/gemini-2.5-flash",      # Family Diversity
    "deepseek_r1": "deepseek/deepseek-r1"           # Reasoning Logic
}
CONCURRENT_PAPERS = 5 
METRIC_KEYS = ['1a_accuracy', '1b_technical_distortion', '2a_novelty_emphasis', 
               '2b_scientific_significance', '3a_engagement_hook_strength', 
               '3b_logical_attractiveness', '3c_call_to_action', '4a_rag_relevance', '4b_rag_utility']


# MODEL_ALIASES = ["gpt4_1", "gpt4o", "gemini2_5_flash", "deepseek_r1_8b", "llama_3_1_8b", "llama_3_1_70b", "deepseek_r1_32b"]
MODEL_ALIASES = ["gpt4_1"]


sem = asyncio.Semaphore(CONCURRENT_PAPERS)
file_lock = asyncio.Lock() # Ensures safe writing to the same file

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


async def save_result_incremental(result, file_path):
    """Appends a single result to the JSON list safely."""
    async with file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except: existing_data = []
        
        existing_data.append(result)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)


async def run_judge(prompt_template, judge_model_id, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            full_prompt = prompt_template.format(**kwargs)
            response = await client.chat.completions.create(
                model=judge_model_id,
                messages=[{"role": "user", "content": full_prompt + "\n\nReturn ONLY JSON."}],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=60.0 
            )
            raw_content = response.choices[0].message.content
            cleaned_json = extract_json(raw_content)
            return EvalResult.model_validate_json(cleaned_json)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
            else:
                return EvalResult(rationale=f"Failed: {str(e)}", score=0)

            
async def evaluate_entry(prompts, row, pbar, model_alias, judge_alias, judge_id, output_json_path):
    async with sem:
        paper_id = row.get('row_index')
        pbar.set_postfix({"paper": paper_id, "judge": judge_alias})
        
        article = row.get('rag_generated_news_article', '')
        ledger = row.get('ledger', {})
        anchor_event = ledger.get('anchor_event', 'N/A')

        tasks = [
            run_judge(prompts[k], judge_id,
                      abstract=row.get('abstract',''), 
                      intro=row.get('introduction',''), 
                      generated_content=article, 
                      pdf_source=row.get('citation',''),
                      anchor_event=str(anchor_event)) 
            for k in METRIC_KEYS
        ]
        
        results = await asyncio.gather(*tasks)
        
        eval_data = {
            "paper_index": paper_id, 
            "generator_alias": model_alias,
            "judge_alias": judge_alias
        }
        for i, key in enumerate(METRIC_KEYS):
            eval_data[f"{key}_score"] = results[i].score
            eval_data[f"{key}_rationale"] = results[i].rationale
        
        await save_result_incremental(eval_data, output_json_path)
        pbar.update(1)
        return eval_data


async def main():
    os.makedirs('ablation_results/evaluations', exist_ok=True)
    
    # Define the 4 Ablation Result Files (should already exist from generation step)
    ABLATION_FILES = [
        "no_connection_rubrics",
        "no_discovery_loop",
        "no_drafting",
        "no_full_scrape", 
        "full_pipeline"
    ]
    
    # Load Metrics Prompts
    loaded_prompts = {}
    for key in METRIC_KEYS:
        path = f"prompts/rag_with_evolving_rubrics/{key}.txt"
        if os.path.exists(path):
            with open(path, 'r') as f:
                loaded_prompts[key] = f.read()
    
    for ablation_alias in ABLATION_FILES:
        input_file = f"ablation_results/results_{ablation_alias}.json"
        
        if not os.path.exists(input_file):
            print(f"⚠️ Skipping {ablation_alias}: Input file not found.")
            continue

        with open(input_file, 'r') as f:
            generated_data = json.load(f)

        for judge_alias, judge_id in JUDGE_MODELS.items():
            output_json = f"ablation_results/evaluations/eval_{ablation_alias}_by_{judge_alias}.json"
            output_csv = f"ablation_results/evaluations/eval_{ablation_alias}_by_{judge_alias}.csv"

            # RESUME LOGIC: Check already evaluated paper_indices
            processed_indices = set()
            if os.path.exists(output_json):
                try:
                    with open(output_json, 'r') as f:
                        existing_evals = json.load(f)
                        processed_indices = {r['paper_index'] for r in existing_evals}
                except Exception:
                    processed_indices = set()

            # Filter for rows that have been generated but not yet evaluated
            # Also filter for row_index < 50 to stay within your ablation sample size
            remaining_data = [
                row for row in generated_data 
                if row.get('row_index') not in processed_indices and row.get('row_index') < 50
            ]

            if not remaining_data:
                print(f"✅ {ablation_alias} is fully evaluated (50/50) by {judge_alias}.")
                continue

            print(f"\n🚀 Resuming Evaluation for {ablation_alias.upper()} | {len(remaining_data)} papers remaining.")

            with tqdm(total=len(remaining_data), desc=f"Eval: {ablation_alias}") as pbar:
                # Use existing evaluate_entry function
                tasks = [
                    evaluate_entry(loaded_prompts, row, pbar, ablation_alias, judge_alias, judge_id, output_json) 
                    for row in remaining_data
                ]
                await asyncio.gather(*tasks)

            # After completion, convert to CSV for easier analysis
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    final_data = json.load(f)
                    pd.DataFrame(final_data).sort_values('paper_index').to_csv(output_csv, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())