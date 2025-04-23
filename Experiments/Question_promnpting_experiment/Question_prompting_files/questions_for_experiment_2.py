
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import json
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util  # Semantic similarity
from tqdm import tqdm
import numpy as np

from rouge import Rouge

import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Load the experiment JSON file
with open("experiment_2.json", "r") as f:
    data = json.load(f)

contexts = data["contexts"]

#---------------------------------------------------------------------------------------------------------------------------------------


models = {
    "Jasu/legalbert": pipeline("question-answering",
        model=AutoModelForQuestionAnswering.from_pretrained("Jasu/bert-finetuned-squad-legalbert"),
        tokenizer=AutoTokenizer.from_pretrained("Jasu/bert-finetuned-squad-legalbert")),
    "deepset/roberta-squad2": pipeline("question-answering",
        model=AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2"),
        tokenizer=AutoTokenizer.from_pretrained("deepset/roberta-base-squad2"))

}


#---------------------------------------------------------------------------------------------------------------------------------------

def compute_em(prediction, gold):
    """
    Compute the Exact Match (EM) score. Returns 1 if prediction equals gold exactly, else 0.
    """
    return 1 if prediction.strip() == gold.strip() else 0

def compute_f1(prediction, gold):
    """
    Compute token-level F1 score for two strings.
    """
    pred_tokens = prediction.strip().split()
    gold_tokens = gold.strip().split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def compute_partial_f1(prediction, gold):
    """
    A partial F1 function can reward partial overlaps even if the token sets are not exactly matching.
    For this example, we define partial F1 as:
         partial_f1 = (number of overlapping tokens) / (average length of gold and prediction)
    """
    pred_tokens = prediction.strip().split()
    gold_tokens = gold.strip().split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0
    avg_length = (len(pred_tokens) + len(gold_tokens)) / 2.0
    return len(common) / avg_length

# Initialize a semantic similarity model from SentenceTransformers

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
def compute_semantic_similarity(prediction, gold):
    """
    Compute the cosine similarity between the embeddings of the prediction and gold answer.
    Returns a float between 0 and 1.
    """
    pred_embedding = semantic_model.encode(prediction)
    gold_embedding = semantic_model.encode(gold)
    cosine_sim = util.cos_sim(pred_embedding, gold_embedding).item()
    return cosine_sim

rouge = Rouge()

# ROUGE Score Calculation
def compute_rouge(prediction, gold):
    scores = rouge.get_scores(prediction, gold)
    return scores[0]['rouge-l']['f']  # Using ROUGE-L F1 score



qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa_pipeline({"question": "What is the capital of France?", "context": "Paris is the capital of France."})
print(result)


#---------------------------------------------------------------------------------------------------------------------------------------


# Run the Experiment

# { model_name: { category: [ { "Format": ..., "Question": ..., "Gold Answer": ..., "Model Answer": ..., metrics... }, ... ] } }

detailed_results = {}

for model_name, qa_model in models.items():
    detailed_results[model_name] = {}

    print(f"Running experiments for model: {model_name}")
    # Loop over each context

    for ctx_item in tqdm(data["contexts"], desc="Contexts"):
        context_text = ctx_item["context"]

        for category_dict in tqdm(ctx_item["Categories"], desc="Categories", leave=False):
            # Each category_dict key is the category name and value is a list of question objects.
            for cat_name, questions in category_dict.items():
                if cat_name not in detailed_results[model_name]:
                    detailed_results[model_name][cat_name] = []

                # Iterate over each question in the category
                for q_obj in tqdm(questions, desc=f"Questions in {cat_name}", leave=False):
                    format_type = q_obj["Format"]
                    question_text = q_obj["Question"]
                    gold_answer = q_obj["Gold Answer"]
                    # Run the QA model with the provided question and context
                    result = qa_model({"question": question_text, "context": context_text})
                    # Some pipelines return a dict; extract the "answer" if available
                    model_answer = result.get("answer", result) if isinstance(result, dict) else result

                    # Compute evaluation metrics
                    em = compute_em(model_answer, gold_answer)
                    f1 = compute_f1(model_answer, gold_answer)
                    partial_f1 = compute_partial_f1(model_answer, gold_answer)
                    rouge_score = compute_rouge(model_answer, gold_answer)
                    semantic_sim = compute_semantic_similarity(model_answer, gold_answer)

                    # Save the detailed result for this question
                    detailed_results[model_name][cat_name].append({
                        "Format": format_type,
                        "Question": question_text,
                        "Gold Answer": gold_answer,
                        "Model Answer": model_answer,
                        "EM": em,
                        "F1": f1,
                        "Partial F1": partial_f1,
                        "ROUGE": rouge_score,
                        "Semantic Similarity": semantic_sim
                    })

# Save detailed results to a JSON file for further analysis
with open("detailed_experiment_2_results.json", "w") as outfile:
    json.dump(detailed_results, outfile, indent=2)


# Aggregate metrics by model and question format (across all contexts and categories)
aggregated_metrics = {}  # { model_name: { format: {metrics} } }
for model_name, cat_results in detailed_results.items():
    format_scores = {}
    for cat, results_list in cat_results.items():
        for entry in results_list:
            fmt = entry["Format"]
            if fmt not in format_scores:
                format_scores[fmt] = {"EM": [], "F1": [], "Partial F1": [], "ROUGE": [], "Semantic Similarity": []}
            format_scores[fmt]["EM"].append(entry["EM"])
            format_scores[fmt]["F1"].append(entry["F1"])
            format_scores[fmt]["Partial F1"].append(entry["Partial F1"])
            format_scores[fmt]["ROUGE"].append(entry["ROUGE"])  
            format_scores[fmt]["Semantic Similarity"].append(entry["Semantic Similarity"])
    aggregated_metrics[model_name] = {}
    for fmt, scores in format_scores.items():
        aggregated_metrics[model_name][fmt] = {
            "Avg EM": np.mean(scores["EM"]),
            "Avg F1": np.mean(scores["F1"]),
            "Avg Partial F1": np.mean(scores["Partial F1"]),
            "Avg ROUGE": np.mean(scores["ROUGE"]),
            "Avg Semantic Similarity": np.mean(scores["Semantic Similarity"])
        }
        
with open("Aggregated_experiment_2_results.json", "w") as outfile:
    json.dump(aggregated_metrics, outfile, indent=2)
