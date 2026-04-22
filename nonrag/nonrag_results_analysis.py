import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import logging

# Setup basic logging to see progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Load data
nonrag_path = 'nonrag/nonrag_evaluation_v6_no_persona.csv'
if not os.path.exists(nonrag_path):
    logging.error(f"File not found: {nonrag_path}")
else:
    df = pd.read_csv(nonrag_path)

    # Identify all individual score columns (excluding the aggregated ones if they exist)
    score_cols = [col for col in df.columns if col.endswith('_score')]

    # Calculate Mean per Model for every individual prompt
    model_individual_summary = df.groupby('generation_model')[score_cols].mean().reset_index()

    # Clean up column names for display (e.g., '1a_accuracy_score' -> '1a Accuracy')
    clean_col_names = {col: col.replace('_score', '').replace('_', ' ').title() for col in score_cols}
    model_individual_summary_clean = model_individual_summary.rename(columns=clean_col_names)
    clean_labels = list(clean_col_names.values())

    print("Non-RAG Individual Metrics Summary:\n", model_individual_summary_clean)

    # Visualization: Detailed Heatmap
    def save_heatmap(data, filename):
        plt.figure(figsize=(12, 6))
        heatmap_data = data.set_index('generation_model')
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5, vmin=1, vmax=5)
        
        plt.title('Non-RAG Individual Prompt Scores by Model', fontsize=15)
        plt.ylabel('Model')
        plt.xlabel('Evaluation Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        logging.info(f"Heatmap saved to {filename}")

    # Visualization: Detailed Radar Chart
    def save_detailed_radar(data, categories, filename):
        labels = np.array(categories)
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for i, row in data.iterrows():
            values = row[categories].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row['generation_model'])
            ax.fill(angles, values, alpha=0.05)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title('Non-RAG Metric Performance Profile', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(filename)
        logging.info(f"Radar chart saved to {filename}")

    # Run Visualizations
    os.makedirs('nonrag', exist_ok=True)
    save_heatmap(model_individual_summary_clean, 'nonrag/nonrag_individual_metrics_heatmap_v6.png')
    save_detailed_radar(model_individual_summary_clean, clean_labels, 'nonrag/nonrag_individual_metrics_radar_v6.png')

    # 6. Export Summary
    model_individual_summary_clean.to_csv('nonrag/nonrag_individual_metrics_summary_v6.csv', index=False)
    logging.info("Non-RAG Analysis complete.")