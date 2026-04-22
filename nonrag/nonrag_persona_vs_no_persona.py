import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df_persona = pd.read_csv('nonrag/nonrag_individual_metrics_summary_v4.csv')
df_no_persona = pd.read_csv('nonrag/nonrag_individual_metrics_summary_v4_no_persona.csv')

# Identify metric columns (all except the model name)
metrics = [col for col in df_persona.columns if col != 'generation_model']

# Calculate Global Average Scores for each model
df_persona['Avg_Score'] = df_persona[metrics].mean(axis=1)
df_no_persona['Avg_Score'] = df_no_persona[metrics].mean(axis=1)

# Merge DataFrames for direct comparison
df_merged = pd.merge(
    df_persona, 
    df_no_persona, 
    on='generation_model', 
    suffixes=('_persona', '_no_persona')
)

# Calculate Score Differences (Delta) for each metric (Positive value means Persona performed better)
for metric in metrics:
    p_col = f'{metric}_persona'
    np_col = f'{metric}_no_persona'
    diff_name = f'{metric}_diff'
    df_merged[diff_name] = df_merged[p_col] - df_merged[np_col]

# Summarise "Wins" per metric
summary_results = []
for metric in metrics:
    diff_col = f'{metric}_diff'
    win_count = (df_merged[diff_col] > 0).sum()
    loss_count = (df_merged[diff_col] < 0).sum()
    tie_count = (df_merged[diff_col] == 0).sum()
    summary_results.append({
        'Metric': metric, 
        'Persona_Wins': win_count, 
        'No_Persona_Wins': loss_count, 
        'Ties': tie_count
    })

df_summary = pd.DataFrame(summary_results)

# Visualise Average Score Comparison
models = df_merged['generation_model'].tolist()
persona_avgs = df_persona['Avg_Score'].tolist()
no_persona_avgs = df_no_persona['Avg_Score'].tolist()

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(12, 7))
plt.bar(x - width/2, persona_avgs, width, label='With Persona', color='#5DADE2', edgecolor='white')
plt.bar(x + width/2, no_persona_avgs, width, label='No Persona', color='#EB984E', edgecolor='white')

plt.xlabel('Generation Model', fontweight='bold')
plt.ylabel('Mean Score (1-5)', fontweight='bold')
plt.title('Comparison: With Persona vs No Persona (Non-RAG)', fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.ylim(0, 5.5)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save outputs
plt.savefig('nonrag/persona_comparison_chart.png')
df_merged.to_csv('nonrag/detailed_persona_comparison_results.csv', index=False)

# Print Summary to Console
print("--- Metric Win Summary ---")
print(df_summary)
print("\n--- Model Average Comparison ---")
print(df_merged[['generation_model', 'Avg_Score_persona', 'Avg_Score_no_persona']])