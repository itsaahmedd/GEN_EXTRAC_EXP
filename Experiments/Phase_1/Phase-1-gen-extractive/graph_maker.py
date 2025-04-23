# Reload necessary libraries after execution state reset
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reload files
baseline_path = "aggregated_cuad_test_results_baseline.json"
mistral_path = "overall_evaluation_results(1).json"
legal_llama_path = "overall_evaluation_results(2).json"

# Load Baseline Extractive Models Results (JSON)
with open(baseline_path, "r") as f:
    baseline_data = json.load(f)

# Load Generative Models Results (JSON)
with open(mistral_path, "r") as f:
    mistral_data = json.load(f)

with open(legal_llama_path, "r") as f:
    legal_llama_data = json.load(f)

# Extractive models (excluding deepset/roberta-squad2)
extractive_models = {
    "Jasu/legalbert": {
        "SemSim": baseline_data["Jasu/legalbert"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["Jasu/legalbert"]["Avg ROUGE"]
    },
    "nlpaueb/legal-bert-base-uncased": {
        "SemSim": baseline_data["nlpaueb/legal-bert-base-uncased"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["nlpaueb/legal-bert-base-uncased"]["Avg ROUGE"]
    },
    "atharvamundada99/bert-large-qa-legal": {
        "SemSim": baseline_data["atharvamundada99/bert-large-qa-legal"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["atharvamundada99/bert-large-qa-legal"]["Avg ROUGE"]
    }
}

# Generative models
generative_models = {
    "Mistral": {
        "SemSim": mistral_data["avg_semantic_similarity"],  
        "ROUGE-L": mistral_data["avg_rouge_l_f1"]
    },
    "Legal LLaMA": {
        "SemSim": legal_llama_data["avg_semantic_similarity"],  
        "ROUGE-L": legal_llama_data["avg_rouge_l_f1"]
    }
}

# Combine models
all_models = {**extractive_models, **generative_models}

# Extract values for plotting
models = list(all_models.keys())
sem_sim_values = [all_models[model]["SemSim"] for model in models]
rouge_values = [all_models[model]["ROUGE-L"] for model in models]

# Plot grouped bar chart
x = np.arange(len(models))  # Model indices
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, sem_sim_values, bar_width, label="Semantic Similarity", color='blue')
plt.bar(x + bar_width/2, rouge_values, bar_width, label="ROUGE-L", color='orange')

# Formatting
plt.xticks(x, models, rotation=45, ha="right")
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Comparison of Extractive and Generative Models (Semantic Similarity & ROUGE-L)")
plt.legend()
plt.tight_layout()
plt.show()
# ----

import json
import matplotlib.pyplot as plt
import numpy as np

# Load Baseline Extractive Models Results (JSON)
with open("aggregated_cuad_test_results_baseline.json", "r") as f:
    baseline_data = json.load(f)

# Load Generative Models Results (JSON)
with open("overall_evaluation_results(1).json", "r") as f:
    mistral_data = json.load(f)

with open("overall_evaluation_results(2).json", "r") as f:
    legal_llama_data = json.load(f)

# Extractive models (excluding deepset/roberta-squad2)
extractive_models = {
    "Jasu/legalbert": {
        "SemSim": baseline_data["Jasu/legalbert"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["Jasu/legalbert"]["Avg ROUGE"]
    },
    "nlpaueb/legal-bert-base-uncased": {
        "SemSim": baseline_data["nlpaueb/legal-bert-base-uncased"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["nlpaueb/legal-bert-base-uncased"]["Avg ROUGE"]
    },
    "atharvamundada99/bert-large-qa-legal": {
        "SemSim": baseline_data["atharvamundada99/bert-large-qa-legal"]["Avg Semantic Similarity"],
        "ROUGE-L": baseline_data["atharvamundada99/bert-large-qa-legal"]["Avg ROUGE"]
    }
}

# Generative models
generative_models = {
    "Mistral": {
        "SemSim": mistral_data["avg_semantic_similarity"],  
        "ROUGE-L": mistral_data["avg_rouge_l_f1"]
    },
    "Legal LLaMA": {
        "SemSim": legal_llama_data["avg_semantic_similarity"],  
        "ROUGE-L": legal_llama_data["avg_rouge_l_f1"]
    }
}

# Combine models
all_models = {**extractive_models, **generative_models}

# Extract values for plotting
models = list(all_models.keys())
sem_sim_values = [all_models[model]["SemSim"] for model in models]
rouge_values = [all_models[model]["ROUGE-L"] for model in models]

# X-axis labels (Semantic Similarity & ROUGE-L)
x_labels = ["Semantic Similarity", "ROUGE-L"]
x = np.arange(len(x_labels))  # Metric indices

# Plot line graph
plt.figure(figsize=(10, 6))

for model in models:
    plt.plot(x, [all_models[model]["SemSim"], all_models[model]["ROUGE-L"]], marker='o', linestyle='-', label=model)

# Formatting the plot
plt.xticks(x, x_labels)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Comparison of Extractive and Generative Models (Semantic Similarity & ROUGE-L)")
plt.legend()
plt.grid(True)
plt.show()
