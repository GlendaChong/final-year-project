import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
FILES_MAP = {
    "Full Pipeline": "ablation_results/evaluations/eval_full_pipeline_by_deepseek_r1.csv",
    "No Connection Rubrics": "ablation_results/evaluations/eval_no_connection_rubrics_by_deepseek_r1.csv",
    "No Discovery Loop": "ablation_results/evaluations/eval_no_discovery_loop_by_deepseek_r1.csv",
    "No Drafting Loop": "ablation_results/evaluations/eval_no_drafting_by_deepseek_r1.csv",
    "No Full Scrape": "ablation_results/evaluations/eval_no_full_scrape_by_deepseek_r1.csv",
}

METRICS = [
    '1a_accuracy_score', '1b_technical_distortion_score', '2a_novelty_emphasis_score',
    '2b_scientific_significance_score', '3a_engagement_hook_strength_score',
    '3b_logical_attractiveness_score', '3c_call_to_action_score',
    '4a_rag_relevance_score', '4b_rag_utility_score'
]

os.makedirs('ablation_results', exist_ok=True)

# --- 2. DATA LOADING & FILTERING (N=50) ---
dfs = {}
for name, fname in FILES_MAP.items():
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        idx_col = 'paper_index' if 'paper_index' in df.columns else 'row_index'
        df.rename(columns={idx_col: 'paper_index'}, inplace=True)
        
        # Using N=50 for higher sensitivity in ablation
        df_50 = df[df['paper_index'] < 50].sort_values(by='paper_index')
        dfs[name] = df_50

full_df = dfs.get("Full Pipeline")
results = []


# --- 3. STATISTICAL ANALYSIS (Paired T-Test + Wilcoxon) ---
for name, df in dfs.items():
    res_row = {'Mode': name, 'Sample_Size': len(df)}
    
    for m in METRICS:
        if m in df.columns:
            res_row[f'{m}_mean'] = df[m].mean()
            
            # Comparison Logic vs Full Pipeline
            if name != "Full Pipeline" and full_df is not None:
                # Merge to ensure we are comparing the same papers (Paired)
                merged = pd.merge(full_df[['paper_index', m]], df[['paper_index', m]], 
                                  on='paper_index', suffixes=('_full', '_ablated'))
                
                if len(merged) > 1:
                    # 1. Paired t-Test (Parametric)
                    _, p_val_t = stats.ttest_rel(merged[m + '_full'], merged[m + '_ablated'])
                    res_row[f'{m}_t_test_p'] = p_val_t
                    
                    # 2. Wilcoxon Signed-Rank Test (Non-Parametric)
                    # Check if differences are all zero to avoid errors
                    diff = merged[m + '_full'] - merged[m + '_ablated']
                    if (diff == 0).all():
                        res_row[f'{m}_wilcoxon_p'] = 1.0
                    else:
                        _, p_val_w = stats.wilcoxon(merged[m + '_full'], merged[m + '_ablated'])
                        res_row[f'{m}_wilcoxon_p'] = p_val_w
        else:
            res_row[f'{m}_mean'] = np.nan
    
    # Calculate Overall Average Score
    valid_means = [res_row[f'{m}_mean'] for m in METRICS if not np.isnan(res_row[f'{m}_mean'])]
    res_row['Average_Score'] = np.mean(valid_means) if valid_means else np.nan
    results.append(res_row)

report_df = pd.DataFrame(results)
report_df.to_csv('ablation_results/ablation_statistical_report_v2.csv', index=False)


# --- 4. VISUALIZATION ---
labels = [m.split('_')[0].upper() for m in METRICS]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for name in dfs.keys():
    mode_data = report_df[report_df['Mode'] == name]
    if not mode_data.empty:
        values = [mode_data[f'{m}_mean'].values[0] for m in METRICS]
        values = [v if not np.isnan(v) else 0 for v in values]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 5)
plt.title("Ablation Study (N=50): Mean Dimension Scores", size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig('ablation_results/ablation_radar_chart_v2.png', bbox_inches='tight')

print("Ablation Analysis Complete with Paired T-Test and Wilcoxon Signed-Rank Test.")