# Reload necessary libraries after execution state reset
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Pre-Fine-Tuning Results (JSON)
pre_finetuning_path = "/mnt/data/aggregated_cuad_test_results_explicit.json"
with open(pre_finetuning_path, "r") as f:
    pre_finetuning_data = json.load(f)

# Convert JSON data to DataFrame where each key is a model
df_pre = pd.DataFrame(pre_finetuning_data).T

# Rename columns for clarity (adding "(Pre)" suffix)
df_pre.rename(columns={
    "Avg EM": "EM (Pre)",
    "Avg F1": "F1 (Pre)",
    "Avg Partial F1": "Partial F1 (Pre)",
    "Avg ROUGE": "ROUGE-L (Pre)",
    "Avg Semantic Similarity": "SemSim (Pre)",
    "Avg Confidence": "Confidence (Pre)"
}, inplace=True)

# Load Post-Fine-Tuning Results (CSV)
df_post_path = "/mnt/data/aggregated_evaluation_results(1).csv"
df_post = pd.read_csv(df_post_path)

# Rename columns for clarity (adding "(Post)" suffix)
df_post.rename(columns={
    "model": "Model",
    "avg_f1": "F1 (Post)",
    "avg_rouge_l_f1": "ROUGE-L (Post)",
    "avg_semantic_similarity": "SemSim (Post)",
}, inplace=True)

# Set the model column as the index for merging
df_post.set_index("Model", inplace=True)

# Standardize model names between JSON and CSV
model_name_mapping = {
    "Jasu/legalbert": "Jasu_bert-finetuned-squad-legalbert",
    "nlpaueb/legal-bert-base-uncased": "nlpaueb_legal-bert-base-uncased",
    "atharvamundada99/bert-large-qa-legal": "atharvamundada99_bert-large-question-answering-finetuned-legal"
}

df_pre = df_pre.rename(index=model_name_mapping)

# Merge Both DataFrames
df_combined = df_pre.join(df_post, how="inner")

# Define metrics to plot
metrics = ["F1", "ROUGE-L", "SemSim", "EM"]

# Filter models for which we have both pre- and post-fine-tuning data
models = df_combined.index.tolist()

# Plot grouped bar charts for each model
for model in models:
    pre_values = [df_combined.loc[model, f"{metric} (Pre)"] if f"{metric} (Pre)" in df_combined.columns else 0 for metric in metrics]
    post_values = [df_combined.loc[model, f"{metric} (Post)"] if f"{metric} (Post)" in df_combined.columns else 0 for metric in metrics]

    x = np.arange(len(metrics))  # Metric indices
    bar_width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - bar_width/2, pre_values, bar_width, label="Pre Fine-Tuning", color='skyblue')
    plt.bar(x + bar_width/2, post_values, bar_width, label="Post Fine-Tuning", color='orange')

    plt.xticks(x, metrics)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(f"Comparison of Pre vs. Post Fine-Tuning for {model}")
    plt.legend()
    plt.tight_layout()
    plt.show()
 