import re, string
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
with open("./experiment_3.json", "r") as f:
    data = json.load(f)

contexts = data["contexts"]

#---------------------------------------------------------------------------------------------------------------------------------------


models = {
    "Jasu/legalbert": pipeline("question-answering",
        model=AutoModelForQuestionAnswering.from_pretrained("Jasu/bert-finetuned-squad-legalbert"),
        tokenizer=AutoTokenizer.from_pretrained("Jasu/bert-finetuned-squad-legalbert"),
        max_answer_len=250,
        ),
    "deepset/roberta-squad2": pipeline("question-answering",
        model=AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2"),
        tokenizer=AutoTokenizer.from_pretrained("deepset/roberta-base-squad2"),
        max_answer_len=250,
        )

}


#---------------------------------------------------------------------------------------------------------------------------------------
# Normalization function to preprocess text before metric comparisons
def normalize_text(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def compute_em(prediction, gold):
    """
    Compute the Exact Match (EM) score. Returns 1 if prediction equals gold exactly, else 0.
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    return 1 if pred_norm == gold_norm else 0

def compute_f1(prediction, gold):
    """
    Compute token-level F1 score for two strings.
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
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
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0
    avg_length = (len(pred_tokens) + len(gold_tokens)) / 2.0
    return len(common) / avg_length

#Initialize a semantic similarity model from SentenceTransformers
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(prediction, gold):
    # Optionally, normalize text for semantic similarity too.
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    pred_embedding = semantic_model.encode(pred_norm)
    gold_embedding = semantic_model.encode(gold_norm)
    cosine_sim = util.cos_sim(pred_embedding, gold_embedding).item()
    return cosine_sim

rouge = Rouge()

def compute_rouge(prediction, gold):
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return 0  # Skip ROUGE calculation for empty normalized text
    try:
        scores = rouge.get_scores(pred_norm, gold_norm)
        return scores[0]['rouge-l']['f']
    except ValueError as e:
        print(f"ROUGE computation failed: {e}")
        return 0  # Assign 0 if ROUGE computation fails


# Maximum BERT input size is 512 tokens; define chunking function with overlap
def chunk_text(text, tokenizer, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        start += max_tokens - overlap  # move the window with overlap
    return chunks


#---------------------------------------------------------------------------------------------------------------------------------------


# Run the Experiment

# { model_name: { category: [ { "Format": ..., "Question": ..., "Gold Answer": ..., "Model Answer": ..., metrics... }, ... ] } }

detailed_results = {}

use_chunking = True  # So we can easily enable and disable chunking


for model_name, qa_model in models.items():
    detailed_results[model_name] = {}

    print(f"Running experiments for model: {model_name}")
    # Loop over each context

    # For chunking, we need access to the tokenizer. Assuming the pipeline exposes it:
    tokenizer = qa_model.tokenizer

    for ctx_item in tqdm(data["contexts"], desc="Contexts"):
        context_text = ctx_item["context"]

     # When using chunking, prepare the list of context chunks
        if use_chunking:
            context_chunks = chunk_text(context_text, tokenizer, max_tokens=512, overlap=50)

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

                    if use_chunking:
                        best_result = None
                        best_score = float('-inf')
                        for chunk in context_chunks:
                            result = qa_model({"question": question_text, "context": chunk})
                            if isinstance(result, dict) and result.get("score", 0) > best_score:
                                best_score = result["score"]
                                best_result = result
                        if best_result:
                            model_answer = best_result.get("answer")
                            confidence = best_result.get("score", None)
                            start_logit = best_result.get("start_logit", None)
                            end_logit = best_result.get("end_logit", None)
                        else:
                            model_answer = ""
                            confidence = None
                            start_logit = None
                            end_logit = None
                    else:
                        result = qa_model({"question": question_text, "context": context_text})
                        model_answer = result.get("answer") if isinstance(result, dict) else result
                        confidence = result.get("score", None)
                        start_logit = result.get("start_logit", None)
                        end_logit = result.get("end_logit", None)


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
                        "Confidence": confidence,
                        "Start Logit": start_logit,
                        "End Logit": end_logit,
                        "EM": em,
                        "F1": f1,
                        "Partial F1": partial_f1,
                        "ROUGE": rouge_score,
                        "Semantic Similarity": semantic_sim
                    })

# Save detailed results to a JSON file for further analysis
with open("detailed_experiment_3_results.json", "w") as outfile:
    json.dump(detailed_results, outfile, indent=2)


# Aggregate metrics by model and question format (across all contexts and categories)
aggregated_metrics = {}  # { model_name: { format: {metrics} } }

for model_name, cat_results in detailed_results.items():
    format_scores = {}
    for cat, results_list in cat_results.items():
        for entry in results_list:
            fmt = entry["Format"]
            if fmt not in format_scores:
                format_scores[fmt] = {"Confidence": [], "EM": [], "F1": [], "Partial F1": [], "ROUGE": [], "Semantic Similarity": []}
            format_scores[fmt]["Confidence"].append(entry["Confidence"])
            format_scores[fmt]["EM"].append(entry["EM"])
            format_scores[fmt]["F1"].append(entry["F1"])
            format_scores[fmt]["Partial F1"].append(entry["Partial F1"])
            format_scores[fmt]["ROUGE"].append(entry["ROUGE"])
            format_scores[fmt]["Semantic Similarity"].append(entry["Semantic Similarity"])
    aggregated_metrics[model_name] = {}
    for fmt, scores in format_scores.items():
        aggregated_metrics[model_name][fmt] = {
            "Avg Confidence": np.mean(scores["Confidence"]),
            "Avg EM": np.mean(scores["EM"]),
            "Avg F1": np.mean(scores["F1"]),
            "Avg Partial F1": np.mean(scores["Partial F1"]),
            "Avg ROUGE": np.mean(scores["ROUGE"]),
            "Avg Semantic Similarity": np.mean(scores["Semantic Similarity"])
        }

with open("Aggregated_experiment_3_results.json", "w") as outfile:
    json.dump(aggregated_metrics, outfile, indent=2)
