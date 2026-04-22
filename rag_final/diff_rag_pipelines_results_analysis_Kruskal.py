import pandas as pd
from scipy import stats
from scipy.stats import rankdata
import scikit_posthocs as sp
import os
import numpy as np

# --- 1. CONFIGURATION ---
files = {
    'Non-RAG Baseline': 'nonrag/nonrag_gpt4_1_baseline_v3.csv',
    'One-Shot Retrieval': 'rag_final/rag_one_shot_evaluation_results_v1.csv',
    'Iterative Retrieval': 'rag_final/rag_iterative_retrieval_evaluation_results_v2.csv',
    'Utility Judgement': 'rag_final/rag_utility_evaluation_results_v1.csv',
    'Evolving-Rubric': 'rag_final/rag_evolving_rubric_evaluation_results_v2.csv'
}

# Phase 1: shared writing metrics (Non-RAG comparable)
phase1_cols = [
    '1a_accuracy_score', '1b_technical_distortion_score',
    '2a_novelty_emphasis_score', '2b_scientific_significance_score',
    '3a_engagement_hook_strength_score', '3b_logical_attractiveness_score',
    '3c_call_to_action_score'
]

# Phase 2: RAG-specific metrics only (architectural differentiators)
phase2_cols = [
    '4a_rag_relevance_score', '4b_rag_utility_score'
]

all_score_cols = phase1_cols + phase2_cols

display_labels = {
    '1a_accuracy_score': '1A',
    '1b_technical_distortion_score': '1B',
    '2a_novelty_emphasis_score': '2A',
    '2b_scientific_significance_score': '2B',
    '3a_engagement_hook_strength_score': '3A',
    '3b_logical_attractiveness_score': '3B',
    '3c_call_to_action_score': '3C',
    '4a_rag_relevance_score': '4A',
    '4b_rag_utility_score': '4B'
}

metric_names = {
    '1a_accuracy_score': 'Accuracy',
    '1b_technical_distortion_score': 'Scientific Nuance',
    '2a_novelty_emphasis_score': 'Novelty',
    '2b_scientific_significance_score': 'Scientific Significance',
    '3a_engagement_hook_strength_score': 'Engagement Hook',
    '3b_logical_attractiveness_score': 'Logical Attractiveness',
    '3c_call_to_action_score': 'Call to Action',
    '4a_rag_relevance_score': 'Contextual Relevance',
    '4b_rag_utility_score': 'Contextual Coherence'
}

# --- 2. DATA LOADING & CLEANING ---
dfs = []
for label, filename in files.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        crit = ['1a_accuracy_score', '1b_technical_distortion_score', '3a_engagement_hook_strength_score']
        df = df[~(df[crit] == 0).any(axis=1)].copy()
        df = df.head(30)

        for col in ['4a_rag_relevance_score', '4b_rag_utility_score']:
            if col not in df.columns or label == 'Non-RAG Baseline':
                df[col] = np.nan
            else:
                df[col] = df[col].replace(0, np.nan)

        df['Configuration'] = label
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['Shared_Avg'] = all_data[phase1_cols].mean(axis=1)
all_data['Overall_Utility'] = all_data[all_score_cols].mean(axis=1)

rag_only = all_data[all_data['Configuration'] != 'Non-RAG Baseline'].copy()

configs_all = list(files.keys())
configs_rag = [c for c in configs_all if c != 'Non-RAG Baseline']


# --- 3. HELPER: Per-metric Kruskal-Wallis + Dunn's post-hoc ---
def per_metric_analysis(data, metrics, group_col, comparisons, phase_label, report_file):
    """
    For each metric independently:
      - Kruskal-Wallis omnibus test
      - Dunn's post-hoc for specified pairwise comparisons
    Holm correction applied within each metric's post-hoc family.
    Avoids aggregating sub-metrics before testing (they are not independent).
    """
    kw_results = []
    dunn_tables = {}

    report_file.write(f"\n{'='*90}\n")
    report_file.write(f"  {phase_label}: PER-METRIC KRUSKAL-WALLIS + DUNN'S POST-HOC\n")
    report_file.write(f"  Rationale: Sub-metrics are not independent within articles, so\n")
    report_file.write(f"  aggregate scores are avoided as test inputs. Each metric tested separately.\n")
    report_file.write(f"{'='*90}\n\n")

    for col in metrics:
        lbl = display_labels[col]
        name = metric_names[col]
        groups = [
            data[data[group_col] == cfg][col].dropna().values
            for cfg in data[group_col].unique()
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        kw_stat, kw_p = stats.kruskal(*groups)
        kw_results.append({
            'Metric': f"{lbl} ({name})",
            'H-statistic': round(kw_stat, 3),
            'p-value': kw_p,
            'Significant': (
                'Yes ***' if kw_p < 0.001 else
                'Yes **'  if kw_p < 0.01  else
                'Yes *'   if kw_p < 0.05  else 'No'
            )
        })

        report_file.write(f"  Metric {lbl}: {name}\n")
        report_file.write(f"  Kruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.4e}\n\n")

        if kw_p < 0.05:
            dunn = sp.posthoc_dunn(data, val_col=col, group_col=group_col, p_adjust='holm')
            dunn_tables[col] = dunn

            data_copy = data[[group_col, col]].dropna().copy()
            data_copy['rank'] = rankdata(data_copy[col])
            mean_ranks = data_copy.groupby(group_col)['rank'].mean()

            report_file.write(f"  Mean Ranks:\n")
            for cfg, rank in mean_ranks.items():
                report_file.write(f"    {cfg:<30}: {rank:.2f}\n")
            report_file.write(f"\n")

            report_file.write(f"  Dunn's Post-hoc (Holm-corrected) -- Selected Comparisons:\n")
            report_file.write(
                f"  {'Group 1':<28} vs {'Group 2':<28} | {'Rank Diff':>10} | {'p-adj':>10} | Reject H0\n"
            )
            report_file.write(f"  {'-'*95}\n")

            for g1, g2 in comparisons:
                if g1 in dunn.index and g2 in dunn.columns:
                    if g1 in mean_ranks.index and g2 in mean_ranks.index:
                        diff = mean_ranks[g2] - mean_ranks[g1]
                        p_val = dunn.loc[g1, g2]
                        sig = (
                            '***' if p_val < 0.001 else
                            '**'  if p_val < 0.01  else
                            '*'   if p_val < 0.05  else 'n.s.'
                        )
                        reject = f"True ({sig})" if p_val < 0.05 else "False"
                        report_file.write(
                            f"  {g1:<28} vs {g2:<28} | {diff:>+10.2f} | {p_val:>10.4f} | {reject}\n"
                        )
            report_file.write(f"\n")
        else:
            report_file.write(f"  No significant variance -- post-hoc not performed.\n\n")

    kw_df = pd.DataFrame(kw_results)
    report_file.write(f"\n  Sensitivity Summary (H-statistics ranked):\n")
    report_file.write(kw_df.sort_values('H-statistic', ascending=False).to_string(index=False))
    report_file.write("\n\n")

    return kw_df, dunn_tables


# --- 4. MEAN SCORES TABLE (descriptive only, not test inputs) ---
summary_means = all_data.groupby('Configuration')[all_score_cols].mean().rename(columns=display_labels)
summary_means['Shared Avg (1A-3C)'] = all_data.groupby('Configuration')['Shared_Avg'].mean()
summary_means['Overall Avg (1A-4B)'] = all_data.groupby('Configuration')['Overall_Utility'].mean()


# --- 5. PHASE 1 COMPARISONS: Non-RAG Baseline vs each RAG strategy ---
phase1_comparisons = [
    ('Non-RAG Baseline', 'One-Shot Retrieval'),
    ('Non-RAG Baseline', 'Iterative Retrieval'),
    ('Non-RAG Baseline', 'Utility Judgement'),
    ('Non-RAG Baseline', 'Evolving-Rubric'),
]

# --- 6. PHASE 2 COMPARISONS: Evolving-Rubric vs other RAG strategies (4A, 4B only) ---
phase2_comparisons = [
    ('Evolving-Rubric', 'One-Shot Retrieval'),
    ('Evolving-Rubric', 'Iterative Retrieval'),
    ('Evolving-Rubric', 'Utility Judgement'),
]


# --- 7. RUN ANALYSIS & SAVE REPORT ---
os.makedirs('rag_final/results_summary', exist_ok=True)
report_path = 'rag_final/results_summary/two_phase_per_metric_report.txt'

with open(report_path, 'w') as f:
    f.write("="*90 + "\n")
    f.write("   TWO-PHASE RAG PERFORMANCE ANALYSIS: PER-METRIC TESTS\n")
    f.write("   Reviewer note addressed: aggregated scores NOT used as test inputs.\n")
    f.write("   Each metric tested independently; Holm correction applied per metric.\n")
    f.write("="*90 + "\n\n")

    # Descriptive table
    f.write("--- DESCRIPTIVE SUMMARY: MEAN SCORES (reporting only, not test inputs) ---\n\n")
    f.write(summary_means.round(3).fillna('--').to_string())
    f.write("\n\n")

    # Phase 1
    kw_p1, dunn_p1 = per_metric_analysis(
        data=all_data,
        metrics=phase1_cols,
        group_col='Configuration',
        comparisons=phase1_comparisons,
        phase_label="PHASE 1: All Configurations vs Non-RAG Baseline (Metrics 1A-3C)",
        report_file=f
    )

    # Phase 2
    kw_p2, dunn_p2 = per_metric_analysis(
        data=rag_only,
        metrics=phase2_cols,
        group_col='Configuration',
        comparisons=phase2_comparisons,
        phase_label="PHASE 2: Evolving-Rubric vs RAG Strategies (Metrics 4A-4B only)",
        report_file=f
    )

    # Grounding-Novelty Trade-off
    f.write("="*90 + "\n")
    f.write("  SUPPLEMENTARY: GROUNDING-NOVELTY TRADE-OFF (2A vs 4A)\n")
    f.write("  Inverse relationship between Novelty framing and Contextual Relevance.\n")
    f.write("="*90 + "\n\n")

    tradeoff_data = rag_only.groupby('Configuration')[
        ['2a_novelty_emphasis_score', '4a_rag_relevance_score']
    ].mean().rename(columns={
        '2a_novelty_emphasis_score': '2A Novelty',
        '4a_rag_relevance_score': '4A Relevance'
    })
    tradeoff_data['Delta (4A - 2A)'] = tradeoff_data['4A Relevance'] - tradeoff_data['2A Novelty']
    f.write(tradeoff_data.round(3).to_string())
    f.write("\n\n")

    # Spearman per configuration
    def spearman_2a_4a(x):
        subset = x[['2a_novelty_emphasis_score', '4a_rag_relevance_score']].dropna()
        if len(subset) > 2:
            return stats.spearmanr(
                subset['2a_novelty_emphasis_score'],
                subset['4a_rag_relevance_score']
            ).statistic
        return np.nan

    spearman_vals = rag_only.groupby('Configuration').apply(
        spearman_2a_4a, include_groups=False
    )
    f.write("  Spearman correlation (2A vs 4A) per configuration:\n")
    f.write(spearman_vals.round(3).to_string())
    f.write("\n\n")


    f.write("="*90 + "\n")
    f.write("  SUPPLEMENTARY: ELITE-TIER CONCENTRATION (Top/Bottom Quartile by Overall Utility)\n")
    f.write("  Descriptive only.\n")
    f.write("="*90 + "\n\n")

    q75 = rag_only['Overall_Utility'].quantile(0.75)
    q25 = rag_only['Overall_Utility'].quantile(0.25)
    top_q = rag_only[rag_only['Overall_Utility'] >= q75]
    bot_q = rag_only[rag_only['Overall_Utility'] <= q25]

    f.write(f"  Top quartile threshold: {q75:.3f} (N={len(top_q)})\n")
    f.write(top_q['Configuration'].value_counts().to_string())
    f.write(f"\n\n  Bottom quartile threshold: {q25:.3f} (N={len(bot_q)})\n")
    f.write(bot_q['Configuration'].value_counts().to_string())
    f.write("\n")

print(f"Report saved to: {report_path}")