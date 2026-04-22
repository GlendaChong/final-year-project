import pandas as pd
import os
from scipy.stats import kruskal, mannwhitneyu

# Configuration
BASE_PATH = 'rag_final/evaluations/'
OUTPUT_DIR = 'rag_final/results_summary/gen_models/'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'model_generalisability_stats.txt')

files = {
    'GPT-4.1': f'{BASE_PATH}eval_gpt4_1.csv',
    'GPT-4o': f'{BASE_PATH}eval_gpt4o.csv',
    'Gemini-2.5-Flash': f'{BASE_PATH}eval_gemini2_5_flash.csv',
    'DeepSeek-R1-32B': f'{BASE_PATH}eval_deepseek_r1_32b.csv',
    'Llama-3.1-70B': f'{BASE_PATH}eval_llama_3_1_70b.csv',
    'Llama-3.1-8B': f'{BASE_PATH}eval_llama_3_1_8b.csv'
}

metrics = [
    '1a_accuracy_score', '1b_technical_distortion_score', '2a_novelty_emphasis_score', 
    '2b_scientific_significance_score', '3a_engagement_hook_strength_score', 
    '3b_logical_attractiveness_score', '3c_call_to_action_score', 
    '4a_rag_relevance_score', '4b_rag_utility_score'
]

# Load and calculate scores
all_data = {}
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['overall_score'] = df[metrics].mean(axis=1)
        all_data[name] = df
    else:
        print(f"Warning: File not found for {name} at {path}")


required_models = ['GPT-4.1', 'Llama-3.1-70B', 'Gemini-2.5-Flash', 'DeepSeek-R1-32B']
if all(m in all_data for m in required_models):
    
    # 1. Metric 4A Variance (RAG Relevance)
    h_4a, p_4a = kruskal(*[all_data[m]['4a_rag_relevance_score'] for m in all_data])
    
    # 2. GPT-4.1 vs Llama-70B (4A - Reasoning Threshold Check)
    _, p_4a_gpt_llama70 = mannwhitneyu(all_data['GPT-4.1']['4a_rag_relevance_score'], 
                                        all_data['Llama-3.1-70B']['4a_rag_relevance_score'])
    
    # 3. Metric 1A Variance (Accuracy)
    h_1a, p_1a = kruskal(*[all_data[m]['1a_accuracy_score'] for m in all_data])
    
    # 4. Gemini vs GPT-4.1 (1A - Truth-Seeker Check)
    _, p_1a_gemini_gpt = mannwhitneyu(all_data['Gemini-2.5-Flash']['1a_accuracy_score'], 
                                       all_data['GPT-4.1']['1a_accuracy_score'])
    
    # 5. DeepSeek-32B vs Llama-70B (Efficiency & Distillation Check)
    _, p_ov_ds_llama70 = mannwhitneyu(all_data['DeepSeek-R1-32B']['overall_score'], 
                                       all_data['Llama-3.1-70B']['overall_score'])
    _, p_4b_ds_llama70 = mannwhitneyu(all_data['DeepSeek-R1-32B']['4b_rag_utility_score'], 
                                       all_data['Llama-3.1-70B']['4b_rag_utility_score'])

    # Construct the formatted output string
    output_text = f"""Cross-Model Statistical Results (N=30 per group)
--------------------------------------------------------------

1. Metric 4A: RAG Relevance
- Kruskal-Wallis H: {h_4a:.2f}
- Kruskal-Wallis p: {p_4a:.4e}
- Pairwise: GPT-4.1 vs. Llama-70B p: {p_4a_gpt_llama70:.4e}

2. Metric 1A: Technical Accuracy
- Kruskal-Wallis H: {h_1a:.2f}
- Kruskal-Wallis p: {p_1a:.4e}
- Pairwise: Gemini-2.5-Flash vs. GPT-4.1 p: {p_1a_gemini_gpt:.4e}

3. Efficiency & Distillation Check (DeepSeek-R1-32B vs. Llama-70B)
- Overall Score p-value: {p_ov_ds_llama70:.4e}
- RAG Utility (4B) p-value: {p_4b_ds_llama70:.4e}

Summary:
- High H-stats indicate significant performance variance across models.
- Low p-values (< 0.05) confirm the 'Reasoning Threshold' and 'Distillation Efficiency' claims.
"""

    # Print and Save
    print(output_text)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output_text)
    print(f"Results saved to: {OUTPUT_FILE}")
else:
    print("Error: Missing one or more required model CSVs to run full statistics.")