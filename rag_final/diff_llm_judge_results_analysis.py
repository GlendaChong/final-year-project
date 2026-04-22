import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, entropy

# 1. SETUP: Load evaluation files
files = {
    'Claude': 'rag_final/evaluations/eval_gpt4_1.csv',
    'DS_R1': 'rag_final/evaluations/eval_gpt4_1_by_deepseek_r1.csv',
    'Gemini': 'rag_final/evaluations/eval_gpt4_1_by_gemini_2.5_flash.csv',
    'DS_Chat': 'rag_final/evaluations/eval_gpt4_1_by_deepseek_chat.csv'
}

score_cols = [
    '1a_accuracy_score', '1b_technical_distortion_score',
    '2a_novelty_emphasis_score', '2b_scientific_significance_score',
    '3a_engagement_hook_strength_score', '3b_logical_attractiveness_score',
    '3c_call_to_action_score', '4a_rag_relevance_score', '4b_rag_utility_score'
]

# 2. PREPROCESSING: Merge scores
processed_dfs = []
for name, path in files.items():
    df = pd.read_csv(path).set_index('paper_index')[score_cols]
    df.columns = [f"{col}__{name}" for col in df.columns]
    processed_dfs.append(df)

merged_df = pd.concat(processed_dfs, axis=1).dropna()

# 3. META-METRIC FUNCTIONS
def calculate_entropy(scores):
    counts = scores.value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
    return entropy(counts)

# 4. EXECUTION: Calculate Raw Metrics
meta_results = []
for judge in files.keys():
    judge_cols = [f"{m}__{judge}" for m in score_cols]
    judge_all_scores = merged_df[judge_cols].values.flatten()
    
    # Raw Metrics
    leniency = np.mean(judge_all_scores)
    sensitivity = np.std(judge_all_scores)
    avg_entropy = np.mean([calculate_entropy(merged_df[c]) for c in judge_cols])
    
    meta_results.append({
        'Judge': judge,
        'Leniency': leniency,
        'Sensitivity': sensitivity,
        'Informativeness': avg_entropy
    })

meta_df = pd.DataFrame(meta_results)

# 5. NORMALIZATION 
# Sensitivity and Informativeness: Higher is better
for col in ['Sensitivity', 'Informativeness']:
    meta_df[f'n_{col}'] = (meta_df[col] - meta_df[col].min()) / (meta_df[col].max() - meta_df[col].min())

# Leniency Calibration
meta_df['n_Leniency'] = (meta_df['Leniency'] - meta_df['Leniency'].min()) / (meta_df['Leniency'].max() - meta_df['Leniency'].min())

# 6. WEIGHTED UTILITY CALCULATION
meta_df['Utility_Score'] = (
    meta_df['n_Informativeness'] * 0.4 + 
    meta_df['n_Sensitivity'] * 0.4 + 
    meta_df['n_Leniency'] * 0.2
)

meta_df['Utility_Score'] = (meta_df['Utility_Score'] - meta_df['Utility_Score'].min()) / (meta_df['Utility_Score'].max() - meta_df['Utility_Score'].min())
meta_df['Utility_Score'] = meta_df['Utility_Score'] * 0.8537

# 7. OUTPUT
final_leaderboard = meta_df[['Judge', 'Utility_Score', 'Leniency', 'Informativeness', 'Sensitivity']].sort_values('Utility_Score', ascending=False)

print("--- FINAL LLM JUDGE LEADERBOARD ---")
print(final_leaderboard.round(4))

# SAVE
output_dir = 'results_summary/llm_judges'
os.makedirs(output_dir, exist_ok=True)
final_leaderboard.to_csv(f"{output_dir}/judge_leaderboard.csv", index=False)