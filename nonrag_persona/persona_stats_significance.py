import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Metrics used for the final score
score_cols = [
    '1a_accuracy_score', '1b_technical_distortion_score',
    '2a_novelty_emphasis_score', '2b_scientific_significance_score',
    '3a_engagement_hook_strength_score', '3b_logical_attractiveness_score',
    '3c_call_to_action_score'
]

def load_data():
    # Load all persona datasets
    df_t1 = pd.read_csv('nonrag/nonrag_evaluation_v6_no_persona.csv')
    df_t1['persona_version'] = 't1'

    df_t2_t3 = pd.read_csv('nonrag_persona/eval_results_t2_vs_t3_w_archetypes.csv')
    df_t2_t3['persona_version'] = df_t2_t3['persona_version'].apply(lambda x: f"t{x}" if str(x).isdigit() else str(x))
    
    df_t4_t5_t6 = pd.read_csv('nonrag_persona/eval_results_t4_vs_t5_vs_t6.csv')
    df_t4_t5_t6['persona_version'] = df_t4_t5_t6['persona_version'].apply(lambda x: f"t{x}" if str(x).isdigit() else str(x))

    # Combine and calculate the Mean Quality Score per row
    combined = pd.concat([df_t1, df_t2_t3, df_t4_t5_t6], ignore_index=True)
    combined['mean_quality_score'] = combined[score_cols].mean(axis=1)
    return combined

def run_tournament(df):
    for model in df['generation_model'].unique():
        print(f"\n{'='*20} MODEL: {model} {'='*20}")
        m_df = df[df['generation_model'] == model]

        # ROUND 1: Compare t1, t2, t3 to find the 'Champion'
        r1_personas = ['t1', 't2', 't3']
        r1_means = m_df[m_df['persona_version'].isin(r1_personas)].groupby('persona_version')['mean_quality_score'].mean()
        champ = r1_means.idxmax()
        
        print(f"\n[Round 1] Baseline (t1) vs Initial Personas:")
        for p in ['t2', 't3']:
            # Align scores by paper_index for a Paired T-Test
            t1_s = m_df[m_df['persona_version'] == 't1'].set_index('paper_index')['mean_quality_score']
            p_s = m_df[m_df['persona_version'] == p].set_index('paper_index')['mean_quality_score']
            
            # Merge to ensure exact paper-to-paper comparison
            merged = pd.concat([t1_s, p_s], axis=1, keys=['t1', 'p']).dropna()
            t_stat, p_val = stats.ttest_rel(merged['t1'], merged['p'])
            diff = merged['p'].mean() - merged['t1'].mean()
            
            sig = "SIGNIFICANT (*)" if p_val < 0.05 else "Not Significant"
            print(f"t1 vs {p}: Diff={diff:+.3f}, p={p_val:.4f} -> {sig}")

        print(f"\n>>> CHAMPION: {champ} (Mean: {r1_means[champ]:.3f})")

        # ROUND 2: Compare Champion against t4, t5, t6
        print(f"\n[Round 2] Champion ({champ}) vs Specialized Personas:")
        for p in ['t4', 't5', 't6']:
            champ_s = m_df[m_df['persona_version'] == champ].set_index('paper_index')['mean_quality_score']
            p_s = m_df[m_df['persona_version'] == p].set_index('paper_index')['mean_quality_score']
            
            merged = pd.merge(champ_s, p_s, on='paper_index', suffixes=('_champ', '_p'))
            t_stat, p_val = stats.ttest_rel(merged['mean_quality_score_champ'], merged['mean_quality_score_p'])
            diff = merged['mean_quality_score_p'].mean() - merged['mean_quality_score_champ'].mean()
            
            sig = "SIGNIFICANT (*)" if p_val < 0.05 else "Not Significant"
            print(f"{champ} vs {p}: Diff={diff:+.3f}, p={p_val:.4f} -> {sig}")

# Execute
combined_df = load_data()
run_tournament(combined_df)