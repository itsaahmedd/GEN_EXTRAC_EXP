import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load Baseline Questions Results (JSON)
baseline_path = "aggregated_cuad_test_results_baseline.json"
with open(baseline_path, "r") as f:
    baseline_data = json.load(f)

# Load Explicit Questions Results (JSON)
explicit_path = "aggregated_cuad_test_results_explicit.json"
with open(explicit_path, "r") as f:
    explicit_data = json.load(f)

# Convert JSON data to DataFrames
df_baseline = pd.DataFrame(baseline_data).T
df_explicit = pd.DataFrame(explicit_data).T

# Rename columns for clarity (adding "(Baseline)" and "(Explicit)")
df_baseline.rename(columns={
    "Avg EM": "EM (Baseline)",
    "Avg F1": "F1 (Baseline)",
    "Avg Partial F1": "Partial F1 (Baseline)",
    "Avg ROUGE": "ROUGE-L (Baseline)",
    "Avg Semantic Similarity": "SemSim (Baseline)",
    "Avg Confidence": "Confidence (Baseline)"
}, inplace=True)

# df_explicit.rename(columns={
#     "Avg EM": "EM (Explicit)",
#     "Avg F1": "F1 (Explicit)",
#     "Avg Partial F1": "Partial F1 (Explicit)",
#     "Avg ROUGE": "ROUGE-L (Explicit)",
#     "Avg Semantic Similarity": "SemSim (Explicit)",
#     "Avg Confidence": "Confidence (Explicit)"
# }, inplace=True)

# # Merge Both DataFrames
# df_comparison = df_baseline.join(df_explicit, how="inner")

# Define metrics to compare
metrics = ["EM", "F1", "ROUGE-L", "SemSim", "Confidence"]

# # Plot grouped bar charts for each model
# for model in df_comparison.index:
#     baseline_values = [df_comparison.loc[model, f"{metric} (Baseline)"] for metric in metrics]
#     explicit_values = [df_comparison.loc[model, f"{metric} (Explicit)"] for metric in metrics]

#     x = np.arange(len(metrics))  # Metric indices
#     bar_width = 0.35

#     plt.figure(figsize=(8, 5))
#     plt.bar(x - bar_width/2, baseline_values, bar_width, label="Baseline Questions", color='blue')
#     plt.bar(x + bar_width/2, explicit_values, bar_width, label="Explicit Questions", color='orange')

#     plt.xticks(x, metrics)
#     plt.xlabel("Metrics")
#     plt.ylabel("Score")
#     plt.title(f"Comparison of Baseline vs. Explicit Questions for {model}")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# ----------- All baseline comparison to pick best models for this experiment:

# Define metrics to compare
metrics = ["EM (Baseline)", "F1 (Baseline)", "ROUGE-L (Baseline)", "SemSim (Baseline)", "Confidence (Baseline)"]

# Extract values for each model
models = df_baseline.index.tolist()
values = {model: df_baseline.loc[model, metrics].tolist() for model in models}

# X-axis labels (metrics)
x_labels = ["EM", "F1", "ROUGE-L", "SemSim", "Confidence"]
x = np.arange(len(x_labels))  # Metric indices

# Plot line graph
plt.figure(figsize=(10, 6))

for model in models:
    plt.plot(x, values[model], marker='o', linestyle='-', label=model)

# Formatting the plot
plt.xticks(x, x_labels)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Comparison of Extractive Models on Baseline Questions")
plt.legend()
plt.grid(True)
plt.show()
