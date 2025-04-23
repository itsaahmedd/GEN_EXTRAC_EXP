# Student Lease AI Bot (Tool) – Experimental Prototype

**Disclaimer:**  
This tool is a demo version and part of testing the results of an experiment in the real world. It is not production-ready and was built as a research prototype to test retrieval-augmented generation (RAG) for answering questions about lease agreements. With additional funding, time, and engineering, this system can be easily scaled into a production-ready solution as the core pillars have been set.

---

## Table of Contents

- [Overview](#overview)
- [Experimental Background & Rationale](#experimental-background--rationale)
- [Key Components & Methodology](#key-components--methodology)
  - [1. PDF Preprocessing and Markdown Extraction](#1-pdf-preprocessing-and-markdown-extraction)
  - [2. Document Splitting and Chunking](#2-document-splitting-and-chunking)
  - [3. Retrieval Strategies](#3-retrieval-strategies)
  - [4. Generative QA with Legal Models](#4-generative-qa-with-legal-models)
  - [5. Conversation Management](#5-conversation-management)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Scalability & Future Improvements](#scalability--future-improvements)
- [License](#license)

---

## Overview

This Student Lease AI Bot is an experimental prototype designed to help students understand their lease agreements by answering questions based on the document's content. The system uses a combination of PDF preprocessing, document chunking, context retrieval, and a retrieval-augmented generative model to provide context-aware responses. The design of this tool is directly informed by experimental findings, ensuring that the optimal preprocessing and retrieval techniques are applied.

Below is a diagram of the pipeline:

```
       +--------------------------------+
       |  Lease Agreement PDF Upload    |
       |    (User provides a PDF)       |
       +---------------+----------------+
                       |
                       v
       +--------------------------------+
       |  PDF Preprocessing &           |
       |  Markdown Extraction           |
       +---------------+----------------+
                       |
                       v
       +--------------------------------+
       |  Document Splitting & Chunking |
       |   (Based on token count)       |
       +---------------+----------------+
                       |
                       v
       +--------------------------------+
       |  Context Retrieval             |
       |  (Using FAISS for semantic     |
       |   matching to extract relevant |
       |   chunks)                      |
       +---------------+----------------+
                       |
                       v
       +--------------------------------+
       |  Generative QA Model           |
       |  (Mistral‑7B Instruct generates  |
       |   concise, context-aware       |
       |   answers)                     |
       +---------------+----------------+
                       |
                       v
       +--------------------------------+
       |  Conversation Management       |
       |  (Chatbot interface & answer   |
       |   generation with history)     |
       +--------------------------------+
```

---

## Experimental Background & Rationale

**Connecting Experiment to Prototype:**  
Based on extensive experiments with various preprocessing, retrieval, and question-phrasing methods (as detailed in the experimental README), this prototype implements the best performing configurations. These experimental insights directly inform the design of our tool, ensuring that the approach yields the most accurate and context-aware responses.

- **Research Focus:**  
This project explores how question formatting, retrieval methods (TF-IDF, BM25, FAISS, etc.), and chunking affect QA performance on lease agreements.
  
- **Generative vs. Extractive Models:**  
Experiments demonstrated that a legal-trained generative model (Mistral‑v2 Instruct) delivers concise and friendly answers when paired with an effective retrieval mechanism (FAISS). This insight has been directly incorporated into our prototype.

- **Document Splitting & Preprocessing:**  
The experimental findings underscored the importance of converting PDFs to Markdown and splitting documents based on header sections to improve context retrieval precision.

---

## Key Components & Methodology

The following components have been implemented based on experimental insights:

### 1. PDF Preprocessing and Markdown Extraction
- **Conversion:**  
  - PDFs are converted to Markdown using `pymupdf4llm` for improved cleaning and logical sectioning.
- **Cleaning:**  
  - Functions remove artifacts and non-ASCII characters.
- **Splitting:**  
  - Markdown text is split by headers to form distinct sections.
- **JSON Save:**  
  - Sections are stored in a JSON file, with each entry containing a "title" and "content".

### 2. Document Splitting and Chunking
- **Token-Based Chunking:**  
  - Long sections are further split into smaller chunks using a Hugging Face tokenizer, ensuring they fit within model input limits.
- **Purpose:**  
  - This approach, validated by our experiments, maintains context integrity and enhances retrieval performance.

### 3. Retrieval Strategies
- **FAISS Retriever:**  
  - Utilizes SentenceTransformer embeddings to index and retrieve semantically relevant chunks, based on the best-performing method from our research.

### 4. Generative QA with Legal Models
- **Model Choice:**  
  - Mistral‑v2 Instruct is employed for its ability to generate concise, context-aware answers.
- **Prompt Construction:**  
  - The prompt integrates:
    - Directives for tone and behavior.
    - Retrieved context.
    - Recent chat history.
    - The user's question.

### 5. Conversation Management
- **History Storage:**  
  - Conversation history is stored per contract to maintain context.
- **Contextual Retrieval:**  
  - Recent conversation messages are retrieved to ensure coherent and human-like responses, a feature highly valued by student feedback.

---

## Project Structure

```
/project-root
│
├── backend
│   ├── main.py                # FastAPI backend for file uploads, context retrieval, conversation management, and answer generation.
│   ├── database.py            # SQLAlchemy configuration and session management.
│   ├── models.py              # SQLAlchemy models (Contract, Conversation).
│   ├── retrievers.py          # Retrieval logic (e.g., FAISSRetriever).
│   └── requirements.txt       # Python dependencies.
│   └── test.db                # SQLite file produced after running main.py.
│
├── frontend
│   ├── src
│   │   ├── App.jsx            # Main React component with routing.
│   │   ├── ChatbotLayout.jsx  # Chat interface with conversation history and PDF preview.
│   │   ├── PDFUpload.jsx      # PDF upload component.
│   │   ├── ListContracts.jsx  # Displays a list of contracts.
│   │   ├── HomePage.jsx       # Landing page.
│   │   ├── NavBar.jsx         # Navigation bar.
│   │   └── theme.js           # Custom Material UI theme.
│   └── package.json           # Frontend dependencies.
│   └── .env                   # Environment variables.
│
└── README.md                  # This file.
```

---

## Prerequisites

Based on our experimental findings, the following setup ensures optimal performance:

- **Node.js & npm:** Install [Node.js](https://nodejs.org/) (v14+ recommended).
- **Python 3.9+**
- **pip:** Python package installer.
- **Database:** SQLite, PostgreSQL, etc. (configured via your backend .env).
- **Optional:** Docker & Docker Compose for containerized deployments.

### Environment Variables

**Frontend (.env):**  
Place a `.env` file at the root of your `frontend` directory (next to `package.json`), and add:

```env
REACT_APP_BACKEND_URL=http://127.0.0.1:8000
REACT_APP_NGROK_URL=https://your-ngrok-url.ngrok.io
```

---

## Installation & Setup

Based on our experiments, following these steps ensures the optimal configuration for this prototype:

### 1. Backend

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**  
   Set up authentication tokens (e.g., for Hugging Face, ngrok) as needed.

5. **Database configuration:**  
   Is `test.db` does not exist run `main.py` to create a database.

### 2. Frontend

1. **Navigate to the student_lease_app in frontend directory:**
   ```bash
   cd frontend
   cd client
   cd student_lease_app
   ```
2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

### 3. Collab (Inference Endpoint)

Due to macOS limitations with the bitsandbytes library (which is used for model quantization), local inference is slow. Therefore, we use a Google Colab instance with GPU acceleration. Below is an excerpt from the Colab notebook:

```python
!pip uninstall -y bitsandbytes
!pip install --no-cache-dir bitsandbytes
!pip install --upgrade accelerate
!pip install fastapi uvicorn nest-asyncio pyngrok transformers sentence-transformers

import nest_asyncio
nest_asyncio.apply()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, Body
from pyngrok import ngrok
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

# Create the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/generate-answer")
async def options_generate_answer():
    return Response(status_code=200)

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI backend!"}

# Global variables for models, tokenizer, and device
embedding_model = None
const_tokenizer = None
const_model = None
const_device = None

def load_models():
    global embedding_model, const_tokenizer, const_model, const_device
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    const_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", const_device)
    const_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    const_tokenizer = AutoTokenizer.from_pretrained(const_model_name, use_auth_token="YOUR_HF_TOKEN")
    const_model = AutoModelForCausalLM.from_pretrained(
        const_model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map={"": const_device},
        use_auth_token="YOUR_HF_TOKEN"
    )
    print("Model loaded.")

@app.on_event("startup")
async def startup_event():
    load_models()

directives = (
    "Your job is to serve as an assistant to help students answer questions about their lease agreement. "
    "Be concise, friendly, and factual. Focus entirely on the document context provided. "
    "If the answer is not clear from the context, ask follow-up questions. "
    "If you still can't find an answer, state that it is not in the context."
)

@app.post("/generate-answer")
async def generate_answer_endpoint(
    question: str = Body(...),
    context: str = Body(...),
    chatHistory: str = Body(...),
):
    prompt = (
        f"{directives}\n\n"
        f"Document Context:\n{context}\n\n"
        f"Conversation History:\n{chatHistory}\n"
        f"---------------------------\n"
        f"Question: {question}\n\n"
        "Final Answer:"
    )
    inputs = const_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2560)
    inputs = {k: v.to(const_device) for k, v in inputs.items()}

    try:
        generated_ids = const_model.generate(**inputs, max_length=2560)
    except Exception as e:
        print(f"Generation error: {e}")
        return {"answer": "Error generating answer."}

    generated_text = const_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Final Answer:" in generated_text:
        answer = generated_text.split("Final Answer:")[-1].strip()
    else:
        answer = generated_text.strip()

    return {"answer": answer}

# Expose via ngrok
if __name__ == "__main__":
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
    public_url = ngrok.connect(8000, "http").public_url
    print("Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Setting up in Colab:
1. Upload the notebook to Google Colab.
2. Set up a runtime with the T4 GPU.
3. Execute the notebook to start the inference endpoint.

---

## Running the Application

### Local Backend

Start the FastAPI backend:
```bash
uvicorn main:app --reload
```
The backend runs on `http://127.0.0.1:8000`.

### React Frontend

Run the frontend in development mode:
```bash
npm start
```
The frontend runs on `http://localhost:3000`.

### Google Colab Inference Server

1. Open and run the provided Colab notebook.
2. Verify that the inference endpoint is exposed via ngrok.
3. Update the frontend configuration (.env file) with the public ngrok URL.

---

## API Endpoints

### File Upload & Processing

- **`POST /upload`**  
  - **Purpose:** Upload a PDF file.  
  - **Process:** The backend converts the PDF to Markdown, preprocesses and chunks the text, saves processed sections as JSON, and creates a new Contract record in the database.  
  - **Request:** Form-data with the PDF file.  
  - **Response:** JSON containing contract id, title, file path, and JSON path.

### Contract Management

- **`GET /contracts`**  
  Retrieve a list of all contracts.
- **`GET /contracts/{contract_id}`**  
  Retrieve a specific contract by its id.
- **`DELETE /contracts`**  
  Delete all contracts.

### Conversation Management

- **`GET /conversations/{contract_id}`**  
  Retrieve conversation history for a given contract.
- **`POST /conversations`**  
  Save conversation history for a given contract.

### Context Retrieval & Answer Generation

- **`POST /retrieve-context`**  
  Retrieve relevant context from the processed PDF sections using FAISS.  
  - **Request:** JSON with `contract_id` and `question`.  
  - **Response:** JSON with the retrieved context.
- **`POST /generate-answer`**  
  Generate an answer using the generative model with the provided question, retrieved context, and recent chat history.  
  - **Request:** JSON with `question`, `context`, and `chatHistory`.  
  - **Response:** JSON with the generated answer.

---

## Troubleshooting

- **Environment Variables:**  
  Ensure your `.env` files are correctly placed and that frontend variables are prefixed with `REACT_APP_`.
- **Restart Servers:**  
  After updating `.env` files, restart both backend and frontend servers.
- **Ngrok/Colab Issues:**  
  If the inference endpoint is unreachable, confirm that the Colab notebook is running, the GPU is active, and the ngrok public URL is updated in your frontend's `.env` file.

---

## Scalability & Future Improvements

**Current Limitations:**

- **Experimental Nature:**  
  This prototype is intended for research and experimentation. It is not optimized for high concurrency or production use. Minimal front-end work was done to focus on gathering student feedback.
- **Local Storage:**  
  Contracts and conversation histories are stored using basic SQLAlchemy and local file storage.
- **Inference Endpoint:**  
  The inference server runs on Google Colab with ngrok, which is not suitable for production.

**Potential for Scaling:**

- **Database & Storage:**  
  Migrate to a distributed database (e.g., PostgreSQL) and cloud-based file storage (e.g., AWS S3).
- **Asynchronous Processing:**  
  Use task queues (e.g., Celery) and caching mechanisms to manage heavy tasks like document indexing and inference.
- **Deployment:**  
  Containerize the application with Docker and orchestrate using Kubernetes for scalability and high availability.
- **Production Readiness:**  
  With additional engineering, security hardening, and performance optimization, the system can evolve into a production-ready application.

---

## License

This project is for academic research and educational use for a final year project. It is not intended for real legal advice.
