import os
os.environ["WANDB_DISABLED"] = "true"

import random
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem, t
import seaborn as sns

import torch
print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback
)

from sentence_transformers import SentenceTransformer, util  # For semantic similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

from datasets import Dataset, DatasetDict
import evaluate

# ========================================================
# 1. Mount Google Drive
# ========================================================
from google.colab import drive
drive.mount('/content/drive')

# ========================================================
# 2. Load dataset 
# ========================================================
dataset_path = "/content/drive/My Drive/Dissertation/cuad_qa_dataset.json"
print("Loading dataset from:", dataset_path)
with open(dataset_path, "r") as f:
    data = json.load(f)

# We'll just take a small subset (500) for demonstration
experiment_data = data["train"][:10]

# Convert dataset to a huggingface Dataset
dataset = Dataset.from_list(experiment_data)
print(dataset)

# Add a unique 'example_id' to each row
example_ids = list(range(len(dataset)))
dataset = dataset.add_column("example_id", example_ids)
print(dataset)

# Shuffle to avoid order bias
dataset = dataset.shuffle(seed=42)

# ========================================================
# 3. Split Dataset
# ========================================================
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
tmp_eval_dataset = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": split_dataset["train"],        # 80%
    "validation": tmp_eval_dataset["train"], # 10%
    "test": tmp_eval_dataset["test"]        # 10%
})

print(dataset_dict)
print("Train size:", len(dataset_dict["train"]))
print("Validation size:", len(dataset_dict["validation"]))
print("Test size:", len(dataset_dict["test"]))


# ========================================================
# 4. Data Preprocessing for QA
# ========================================================


max_length = 512  
doc_stride = 128  


def preprocess_training_examples(example):
    """
    Tokenizes a QA example into chunks while adjusting answer positions.
    The answer_start in the JSON refers to the full context, so we check
    whether the answer appears in each chunk (using offset mappings)
    and adjust indices accordingly.
    """
    question = example["question"].strip()
    context = example["context"]

    # Extract answer information if available.
    if (not example["answers"]["text"] or not example["answers"]["answer_start"] or 
        example["answers"]["answer_start"][0] == -1):
        answer_text = ""
        answer_start = None  # Indicates no answer.
    else:
        answer_text = example["answers"]["text"][0]
        answer_start = example["answers"]["answer_start"][0]

    # Tokenize with sliding window over the context.
    tokenized_examples = tokenizer(
        question,
        context,
        truncation="only_second",  # Only truncate the context.
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=False
    )

    # Get the mapping from each chunk back to the original example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    print(f"[DEBUG] Number of chunks: {len(tokenized_examples['input_ids'])}")
    print("First chunk offset mapping:", offset_mapping[0])

    # Prepare lists to hold the processed fields.
    input_ids_chunks = []
    attention_mask_chunks = []
    start_positions = []
    end_positions = []
    example_ids = []

    # Process each chunk.
    for i, offsets in enumerate(offset_mapping):
        # Identify the index for the [CLS] token.
        cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)

        # Use sequence_ids to determine which tokens belong to the context.
        # If not available, we assume tokens with offset (0,0) belong to the question.
        sequence_ids = (tokenized_examples.sequence_ids(i)
                        if hasattr(tokenized_examples, "sequence_ids")
                        else [0 if off == (0, 0) else 1 for off in offsets])
        # Find boundaries for context tokens.
        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(sequence_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Default positions (no answer in this chunk) are set to the CLS token.
        if answer_start is None or not (offsets[token_start_index][0] <= answer_start and 
                                        offsets[token_end_index][1] >= answer_start + len(answer_text)):
            start_pos = cls_index
            end_pos = cls_index
        else:
            # Adjust the answer positions relative to the chunk.
            # Find the token index where the answer starts.
            start_pos = token_start_index
            while (start_pos < len(offsets) and offsets[start_pos] is not None and 
                   offsets[start_pos][0] <= answer_start):
                start_pos += 1
            start_pos -= 1

            # Find the token index where the answer ends.
            answer_end = answer_start + len(answer_text)
            end_pos = token_end_index
            while (end_pos >= 0 and offsets[end_pos] is not None and 
                   offsets[end_pos][1] >= answer_end):
                end_pos -= 1
            end_pos += 1

        print(f"[DEBUG] Chunk {i}: start_pos = {start_pos}, end_pos = {end_pos}")


        # Save the processed fields.
        input_ids_chunks.append(tokenized_examples["input_ids"][i])
        attention_mask_chunks.append(tokenized_examples["attention_mask"][i])
        start_positions.append(start_pos)
        end_positions.append(end_pos)
        example_ids.append(int(example["example_id"]))



    return {
        "input_ids": input_ids_chunks,
        "attention_mask": attention_mask_chunks,
        "start_positions": start_positions,
        "end_positions": end_positions,
        "example_id": example_ids
    }



def prepare_dataset_for_model(model_name_or_path: str, dataset_dict: DatasetDict) -> DatasetDict:
    # load tokenizer globally so we can set e.g. return_token_type_ids=False
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # We'll remove only the old columns: "id","title","context","question","answers"
    # We'll keep our newly created columns: start_positions, end_positions, example_id

    train_dataset = dataset_dict["train"].map(
        preprocess_training_examples,
        batched=False,
        remove_columns=["id","title","context","question","answers"]
    )

    debug_dataset(train_dataset)


    val_dataset = dataset_dict["validation"].map(
        preprocess_training_examples,
        batched=False,
        remove_columns=["id","title","context","question","answers"]
    )

    test_dataset = dataset_dict["test"].map(
        preprocess_training_examples,
        batched=False,
        remove_columns=["id","title","context","question","answers"]
    )
    
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


# ========================================================
# 5. Metrics & Helpers
# ========================================================

squad_metric = evaluate.load("squad_v2")
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # for semantic similarity
rouge_metric = Rouge()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions_and_confidence(model, tokenizer, dataset):
    """
    Generate chunk-level predictions (start/end logits) for the given dataset,
    returning one dict per chunk that includes 'example_id' & predicted text.
    """
    model.eval()
    model.to(device)
    predictions = []
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=16, 
                                             collate_fn=default_data_collator)
    
    for batch in tqdm(dataloader, desc="Generating predictions"):


        print("[DEBUG] Processing batch with", batch["input_ids"].size(0), "chunks")

        # Ensure all batch items are tensors on device
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                batch[k] = torch.tensor(v)
            batch[k] = batch[k].to(device)
        
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits   = outputs.end_logits
        
        # For each chunk in this batch
        for i in range(start_logits.size(0)):
            start_idx = torch.argmax(start_logits[i]).item()
            end_idx   = torch.argmax(end_logits[i]).item()
            start_conf = float(torch.max(start_logits[i]).item())
            end_conf   = float(torch.max(end_logits[i]).item())
            confidence = (start_conf + end_conf) / 2.0
            
            input_ids_chunk = batch["input_ids"][i]
            answer_ids = input_ids_chunk[start_idx : end_idx+1]
            pred_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)


            print(f"[DEBUG] Prediction for chunk: start_idx={start_idx}, end_idx={end_idx}, confidence={confidence}")
            print("Decoded answer:", pred_answer)

            
            ex_id = batch["example_id"][i].item()
            predictions.append({
                "example_id": ex_id,
                "prediction_text": pred_answer.strip(),
                "confidence": confidence,
                "start_idx": start_idx,
                "end_idx": end_idx
            })


    return predictions

from collections import defaultdict

def aggregate_predictions(predictions):

    # Group chunk-level predictions by example_id and pick highest-confidence
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p["example_id"]].append(p)
    
    final_dict = {}

    for ex_id, chunk_preds in grouped.items():

        best = max(chunk_preds, key=lambda x: x["confidence"])
        final_dict[ex_id] = {
            "prediction_text": best["prediction_text"],
            "confidence": best["confidence"]
        }


        print(f"[DEBUG] example_id {ex_id} has {len(chunk_preds)} chunk predictions")

        
    return final_dict


def compute_f1(prediction: str, ground_truth: str) -> float:
    """A simple token-based F1 for predicted vs. gold text."""
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec  = len(common) / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def evaluate_predictions(predictions, dataset):
    """
    Compare final predictions with gold answers in the original dataset,
    computing token-based F1, S-BERT similarity, ROUGE-L, etc.
    """
    labels = []
    confidences = []
    f1_scores = []
    semantic_scores = []
    rouge_scores = []
    
    for i, p in enumerate(predictions):
        pred_text = p["prediction_text"]
        conf      = p["confidence"]
        
        # The gold answers are still in dataset[i]["answers"]["text"]
        gold_list = dataset[i]["answers"]["text"]
        gold_text = gold_list[0].strip() if gold_list else ""
        
        # Token-based F1
        f1_ = compute_f1(pred_text, gold_text)
        f1_scores.append(f1_)
        
        # 0/1 label if F1 >= 0.6
        label = 1 if f1_ >= 0.6 else 0
        labels.append(label)
        confidences.append(conf)
        
        # S-BERT semantic similarity
        emb_gold = sbert_model.encode(gold_text, convert_to_tensor=True)
        emb_pred = sbert_model.encode(pred_text, convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(emb_gold, emb_pred).item())
        semantic_scores.append(sim)
        
        # ROUGE-L
        if not pred_text.strip() or not gold_text.strip():
            rouge_scores.append(0.0)
        else:
            r = rouge_metric.get_scores(pred_text, gold_text)
            rouge_l_f1 = r[0]["rouge-l"]["f"]
            rouge_scores.append(rouge_l_f1)
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(labels, confidences)
    ap_score = average_precision_score(labels, confidences)
    
    # Find the maximum precision at recall >= 0.8
    p_at_80 = 0.0
    for p, r in zip(precision, recall):
        if r >= 0.8 and p > p_at_80:
            p_at_80 = p
    
    results = {
        "avg_f1": float(np.mean(f1_scores)),
        "avg_semantic_similarity": float(np.mean(semantic_scores)),
        "avg_rouge_l_f1": float(np.mean(rouge_scores)),
        "AUPR": float(ap_score),
        "precision_at_80%": float(p_at_80)
    }
    return results

# ========================================================
# 6. Training &  Evaluation
# ========================================================
from transformers import EvalPrediction

def compute_metrics(p: EvalPrediction):
    return {}

def train_and_evaluate_model(
    model_ckpt: str,
    dataset_dict: DatasetDict,
    out_dir: str = "/content/drive/My Drive/Dissertation/model_output",
    epochs: int=2,
    lr: float=5e-5,
    train_bs: int=32,
    eval_bs: int=32,
    fp16=True
):
    # 1) Tokenize & chunk
    processed_ds = prepare_dataset_for_model(model_ckpt, dataset_dict)
    
    # 2) Load QA model
    config = AutoConfig.from_pretrained(model_ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt, config=config)
    if torch.cuda.is_available():
        model.cuda()
    
    # 3) Training args
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        fp16=fp16,
        save_strategy="steps",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        push_to_hub=False,
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_ds["train"],
        eval_dataset=processed_ds["validation"],
        tokenizer=tokenizer,
        data_collator=custom_data_collator,  # Use our custom collator here.
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 4) Train
    trainer.train()
    
    return {"trainer": trainer, "dataset": processed_ds}





def debug_dataset(ds):
    print("DEBUGGGGGG TIME")
    for i in range(len(ds)):
        ex = ds[i]
        print(f"Example {i} - Num chunks: {len(ex['input_ids'])}")
        # Optionally, print more details for one example:
        if i < 2:
            print(f"start_positions: {ex['start_positions']}")
            print(f"end_positions: {ex['end_positions']}")


def custom_data_collator(features):
    # First, figure out the maximum number of chunks among the features.
    max_chunks = max(len(feature["input_ids"]) for feature in features)
    
    padded_features = []
    for feature in features:
        num_chunks = len(feature["input_ids"])
        pad_len = max_chunks - num_chunks
        
        # For each field, pad with a “dummy” value.
        if num_chunks > 0:
            token_length = len(feature["input_ids"][0])
        else:
            token_length = max_length  # fallback value
        
        # Create padding for each field.
        pad_input_ids = [[0] * token_length] * pad_len
        pad_attention_mask = [[0] * token_length] * pad_len
        pad_start_positions = [0] * pad_len
        pad_end_positions = [0] * pad_len
        
        # Handle example_id robustly.
        if "example_id" in feature:
            ex_id = feature["example_id"] if isinstance(feature["example_id"], list) else [feature["example_id"]]
            pad_example_ids = [ex_id[0]] * pad_len
            new_example_ids = ex_id + pad_example_ids
        else:
            new_example_ids = None

        new_feature = {
            "input_ids": feature["input_ids"] + pad_input_ids,
            "attention_mask": feature["attention_mask"] + pad_attention_mask,
            "start_positions": feature["start_positions"] + pad_start_positions,
            "end_positions": feature["end_positions"] + pad_end_positions
        }
        if new_example_ids is not None:
            new_feature["example_id"] = new_example_ids

        padded_features.append(new_feature)
    
    # Stack each field into a tensor.
    input_ids = torch.tensor([f["input_ids"] for f in padded_features])           # shape: (B, num_chunks, seq_length)
    attention_mask = torch.tensor([f["attention_mask"] for f in padded_features])     # shape: (B, num_chunks, seq_length)
    start_positions = torch.tensor([f["start_positions"] for f in padded_features])   # shape: (B, num_chunks)
    end_positions = torch.tensor([f["end_positions"] for f in padded_features])       # shape: (B, num_chunks)
    
    if "example_id" in padded_features[0]:
        example_ids = torch.tensor([f["example_id"] for f in padded_features])      # shape: (B, num_chunks)
    else:
        example_ids = None

    # Flatten the batch and chunk dimensions.
    B, num_chunks, seq_len = input_ids.shape
    input_ids = input_ids.view(B * num_chunks, seq_len)
    attention_mask = attention_mask.view(B * num_chunks, seq_len)
    start_positions = start_positions.view(B * num_chunks)
    end_positions = end_positions.view(B * num_chunks)
    if example_ids is not None:
        example_ids = example_ids.view(B * num_chunks)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions
    }
    if example_ids is not None:
        batch["example_id"] = example_ids

    return batch



# ========================================================
# 7. Main
# ========================================================
model_candidates = [
    "Jasu/bert-finetuned-squad-legalbert",
    "nlpaueb/legal-bert-base-uncased",
    "atharvamundada99/bert-large-question-answering-finetuned-legal",
    "microsoft/deberta-base",
    "facebook/bart-base"
]


def main():

    all_results = []

    for model_ckpt in model_candidates:

        print(f"\n***** TRAINING {model_ckpt} *****\n")
        out_dir = f"./{model_ckpt.replace('/', '_')}"
        
        # Train the model and get processed datasets.
        artifacts = train_and_evaluate_model(
            model_ckpt,
            dataset_dict,
            out_dir
        )
        
        trainer = artifacts["trainer"]
        processed_ds = artifacts["dataset"]
        test_ds = processed_ds["test"]  # processed, tokenized test
        
        # Inference on chunk-level test data.
        print(f"Generating predictions on test set for {model_ckpt}...")
        chunk_preds = get_predictions_and_confidence(trainer.model, tokenizer, test_ds)
        
        # Aggregate predictions across chunks by example_id.
        aggregated = aggregate_predictions(chunk_preds)
        
        # Reconstruct final predictions for the original examples.
        # We use the original test split (which still contains 'answers') for evaluation.
        orig_test = dataset_dict["test"]
        final_preds = []

        final_preds = []
        for example in orig_test:
            ex_id = example["example_id"]
            pred_data = aggregated.get(ex_id, {"prediction_text": "", "confidence": 0.0})
            final_preds.append({
                "prediction_text": pred_data["prediction_text"],
                "confidence": pred_data["confidence"]
            })

                
        # Evaluate predictions against the gold answers.
        metrics = evaluate_predictions(final_preds, orig_test)
        metrics["model"] = model_ckpt
        all_results.append(metrics)
        print("\nMetrics for", model_ckpt, ":", metrics)
    
    df = pd.DataFrame(all_results)
    print("\n=== Final Results Across All Models ===\n", df)

if __name__ == "__main__":
    main()
