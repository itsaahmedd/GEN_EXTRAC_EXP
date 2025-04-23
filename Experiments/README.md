# Legal Contract QA: Exploring and comparing Extractive and Generative Models for Student Lease QA

# Table of Contents

- [Project Overview](#project-overview)
- [Environment & Setup](#environment--setup)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Phase 1: CUAD-Based Experiments](#phase-1-cuad-based-experiments)
  - [Dataset: CUAD](#dataset-cuad)
  - [1. Question Prompting Sub-Experiment](#1-question-prompting-sub-experiment)
  - [2. Extractive QA - Baseline](#2-extractive-qa---baseline)
  - [3. Extractive QA - Fine-Tuned](#3-extractive-qa---fine-tuned)
  - [4. Generative QA](#4-generative-qa)
- [Phase 2: Real Student Lease Evaluation](#phase-2-real-student-lease-evaluation)
  - [Dataset, Formats & Annotation](#large-manual-annoated-dataset)
  - [PDF Preprocessing](#pdf-preprocessing)
    - [Document Splitting](#document-splitting)
    - [Conversion Strategy](#conversion-strategy)
    - [Extraction and Annotation](#extraction-and-annotation)
  - [1. Extractive QA](#1-extractive-qa)
  - [2. Generative QA](#2-generative-qa)
- [Helper Scripts](#helper-scripts)
- [Final Application Goal](#final-application-goal)
- [Citation & Acknowledgements](#citation--acknowledgements)
- [Disclaimer](#disclaimer)

---

## Project Overview

This project consists of many scripts that evaluate and compare **extractive** and **generative** question-answering (QA) models for understanding legal contracts, with a final goal of building an AI system that helps students query lease agreements. The experiment is broken down into multiple structured phases, culminating in the creation of a domain-specific QA tool using real student leases.

---

## Environment & Setup

- Ensure you have **Python 3.9+** installed.
- Ensure you have access to **Google Collab (T4 GPU)**.
- Ensure you have a valid **Hugging Face token** to access certain models and datasets.
- All scripts in this project were executed using Google Collab, utilising the T4 GPU for accelerated compute performance. **All necessary dependencies are installed via `pip install` in the first code section of every notebook**. This ensures that the code runs smoothly in the Collab environment with the appropriate versions of required libraries and the necessary credentials for Hugging Face services.

---

## Problem Statement

To determine which QA approach (**Extractive or Generative**) and which retrieval method (**Tf-IDF, BM25, Woosh, Fiass**) is most suitable for answering legal questions from student lease agreements, while considering factors such as:

- Comprehension of legal phrasing (most important)
- Mitigation of hallucination risks using RAG
- Limitations related to context length
- Preprocessing PDFs
- Question and prompting styles and its effect on model answers

---

## Project Structure

```
.
├── Helper Scripts                  # Utility scripts (For loading CUAD)
├── Other                           # Miscellaneous files - initial results before running Phase 1 for extarctive models
├── Phase_1                         # Experiments on the CUAD dataset
│   ├── Phase_1_extractive_scripts        # Extractive models, zero-shot
│   ├── Phase_1_extractive_results        # Results 
│   ├── Phase_1_finetuned_extractive_scripts # Extractive models after fine-tuning
│   ├── Phase_1_finetuned_extractive_results # Results 
│   ├── Phase_1_gen                        # Generative models on CUAD 
│   ├── Phase_1_gen_result                 # Results 
│   └── Phase-1-gen-extractive             # Cross-checks
├── Phase_2                         # Experiments on manually annotated and collected student lease dataset
│   ├── Extractive                        # Top-3 extractive models tested on leases on 4 different retreiaval methods
│   ├── Generative                        # Mistral & Legal-LLaMA (fiasss and hybrid retrieval approach)
│   ├── agreements                        # Original lease PDFs + QA versions + markdown
│   └── processed                         # Sectioned JSONs of lease texts
├── Question_prompting_data        # Muanually annotated dataset for sub-experiment on question phrasing
├── Question_prompting_files       
├── Question_prompting_notebooks   # Notebooks for running QA with different formats
└── Question_prompting_results     # Metrics and performance breakdown by format
```

---

## Phase 1: CUAD-Based Experiments

### Dataset: CUAD (Contract Understanding Atticus Dataset)

- Publicly available dataset for legal QA.
- Used to stress-test various QA models in a slightly different domain (commercial contracts).

### 1. Question Prompting Sub-Experiment

**Folder:** `Question_prompting_experiment`

- **Goal:** Identify the best phrasing strategy for extractive QA models before running the Phase 1 scripts.
- **Annotation:** Manual annotation done (~10 hours) to create `experiment_1.json`, `experiment_2.json`,`experiment_3.json` in the **Folder:** `Question_prompting_files`
- **Variants tested:** **Baseline**, **Simplified**, **Explicit**, **Yes/No**.
- **Outcome:** Baseline-style, concise questions perform best, and the output is used to standardize CUAD and lease QA experiment setups.

### 2. Extractive QA - Baseline

**Folder:** `Phase_1_extractive_scripts`

- **Models tested:** BERT, RoBERTa, LegalBERT, DeBERTa.
- **Dataset:** Evaluated on full CUAD (2000 test entries).
- **Results:** Stored in `Phase_1_extractive_results`.

### 3. Extractive QA - Fine-Tuned

**Folder:** `Phase_1_finetuned_extractive_scripts`

- **Approach:** Fine-tuned using the CUAD training set (3430 train entries).
- **Setup:** Used the same questions and format as in explicit (not baseline) to explore if fine tuning can have a greater effect than question format.
- **Results:** Stored in `Phase_1_finetuned_extractive_results`.

### 4. Generative QA

**Folder:** `Phase_1_gen`

- **Models:** Falcon-7B, Mistral 7B Instruct, Legal-LLaMA.
- **Dataset constraints:** Due to compute/time constraints:
  - Only **200 test entries** were used for Mistral 7B Instruct v0.2 and Legal-LLaMA.
  - Only **50 test entries** were used for Falcon-7B.
- **Evaluation:** Based on semantic similarity, BERTScore, and ROUGE.

---

## Phase 2: Real Student Lease Evaluation

### Large Manual Annoated Dataset


**Folder:** `Phase_2`

- **Dataset:** 18 student lease agreements collected from students within the university.
- **Data Formats:** Split into PDFs, Markdown versions, QA-enhanced PDFs, and sectioned JSONs.
- **Annotation:** Manual annotation done (~50 hours) to create `gold_standard.json` under the supervision of legal students, **created 512 questions and answers**

### PDF Preprocessing

Handling PDF documents introduces unique challenges compared to more structured formats like HTML or Markdown. The preprocessing pipeline for PDFs involves:

- **Document Splitting:**
  - **Handling Variable Lengths:** Real-world PDFs vary in length; splitting helps maintain consistent processing.
  - **Overcoming Model Limitations:** Many embedding and language models have input size constraints. Splitting PDFs into manageable sections of `400 tokens` (to allow room for special and question tokens) prevents exceeding these limits.
  - **Improving Representation Quality:** Shorter, focused text chunks yield more accurate embeddings.
  - **Enhancing Retrieval Precision:** Finer granularity in splitting leads to more precise matching of queries to relevant sections.
  - **Optimizing Computational Resources:** Processing smaller text chunks is more memory efficient and allows for parallel processing.

- **Conversion Strategy:**
  - Due to the semi-structured nature of PDFs (with formatting cues like bold or underlined text indicating titles), direct extraction can be challenging. Many tenancy agreements contain tables and regular text that require different extraction methods.
  - Initial attempts involved converting PDFs to HTML; however, this often resulted in loss of structure (e.g., text in spans and divs).
  - A more effective approach was converting PDFs to Markdown, which better preserves headers and sectioning, improving subsequent text processing and retrieval.

- **Extraction and Annotation:**
  - I use `pymupdf4llm` for text extraction into Markdown.
  - Regular expressions and text-parsing strategies are applied to extract Q&A pairs from the documents.
  - The processed text is then split, annotated, and stored in JSON format for further evaluation against the manually created gold standard.

### 1. Extractive QA

**Folder:** `Phase_2/Extractive`

- **Models:** Top 3 extractive models from Phase 1.
- **Testing:** Applied on all 18 leases.
- **Methods:** Tested on 4 different retrieval methods (**TF-IDF, BM25, Woosh, FIASS**).
- **Evaluation:** Results compared against `gold_standard.json`.
- **Results:** Stored in the same folder.

### 2. Generative QA

**Folder:** `Phase_2/Generative`

- **Models:** Mistral 7B Instruct and Legal-LLaMA.
- **Testing:** 
  - Mistral tested on all 18 leases.
  - Legal-LLaMA tested on the first 10 leases (due to time/compute limits).
- **Methods:** Tested on 2 different retrieval methods (**FIASS, Hybrid-Approach (BM25 & FIASS)**).
- **Evaluation:** Results compared against `gold_standard.json`.
- **Results:** Stored in the same folder.


---

## Helper Scripts

- `dataset_loader.py` (in `Helper Scripts`) is used to load the CUAD dataset efficiently due to its size.

---

## Final Application Goal

Develop a student-facing tool that serves as a real-world testing environment for the best performing QA model. This tool integrates the optimal combination of preprocessing, retrieval, PDF extraction, and question phrasing methods—determined through our experiments—to allow students to upload a lease document and ask legal questions. The tool enables direct student feedback on the system’s performance.

- **Backend:** RAG-based generative model using Mistral.
- **Frontend:** React.
- **Approach Informed by All Phases:**
  - Utilise **baseline question phrasing**.
  - Apply **Fiass retrieval** to select the most relevant clauses.
  - Leverage optimized PDF preprocessing and extraction techniques.
  - Generate answers by either extracting text spans or producing summaries based on the best performing configuration.

**Access the tool in the tool directory of the folder.**

---

## Citation & Acknowledgements

- CUAD Dataset: [CUAD GitHub](https://github.com/TheAtticusProject/cuad)

---

## Disclaimer

This project is for academic research and educational use for a final year project. Not to be used for real legal advice.
