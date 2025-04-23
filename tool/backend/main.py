# Remove or comment out the module-level FAISS import:
# import faiss
import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pymupdf4llm
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Contract, Base
from pydantic import BaseModel
from fastapi import Response, Body
from retrievers import FaissRetriever
# For TF-IDF retrieval:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import HTTPException
from pydantic import BaseModel

from transformers import AutoTokenizer


MAX_CHUNK_TOKENS = 400

# Choose a tokenizer for counting tokens (using a BERT tokenizer as an example)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()


from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Enable CORS for your React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally, add an explicit OPTIONS handler for the generate-answer endpoint:
@app.options("/generate-answer")
async def options_generate_answer():
    return Response(status_code=200)



# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def clean_text(text: str) -> str:

    """
    Remove unwanted artifacts and non-ASCII characters.
    """

    text = re.sub(r'â', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'â*', '', text)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    return text.strip()

def split_markdown_by_headers(markdown: str):

    """
    Splits markdown text into sections based on headers.
    Returns a list of dicts with keys 'title' and 'content'.
    """

    sections = []
    current_section = {"title": None, "content": ""}

    for line in markdown.splitlines():

        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            if current_section["title"] is not None or current_section["content"].strip():
                sections.append(current_section)
            title = header_match.group(2).strip()
            current_section = {"title": title, "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["title"] is not None or current_section["content"].strip():
        sections.append(current_section)

    return sections

def process_content(text: str) -> str:

    """
    Replace newline characters with a space and collapse extra spaces.
    """

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_markdown_file(markdown):

    """
    Process a markdown file: clean, split by headers, process content,
    and filter out trivial sections.
    """

   
    markdown = clean_text(markdown)
    sections = split_markdown_by_headers(markdown)

    for sec in sections:
        sec["content"] = process_content(sec["content"])

    # Filter out sections that are too trivial
    filtered_sections = []

    for sec in sections:

        if not sec["content"].strip():
            continue
        filtered_sections.append(sec)

    return filtered_sections




def chunk_section_by_tokens(section, max_tokens: int = MAX_CHUNK_TOKENS):

    """
    Use the Hugging Face tokenizer to count tokens and split a section's content into sub‐chunks.
    The method splits on sentence boundaries if possible.
    """

    text = section["content"]

    # Tokenize using the model's tokenizer (which returns token IDs)
    tokens = tokenizer.tokenize(text)

    if len(tokens) <= max_tokens:
        return [section]

    # For a better split, we can try to split by sentences.
    # Here we use a naive regex sentence split; you might also use nltk.sent_tokenize.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_tokens = []

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)

        # If adding the sentence exceeds max_tokens, store the current chunk.
        if len(current_tokens) + len(sent_tokens) > max_tokens:
            if current_chunk:
                chunks.append({
                    "title": section["title"],
                    "content": current_chunk.strip()
                })
            # Start a new chunk with this sentence.
            current_chunk = sent + " "
            current_tokens = sent_tokens
        else:
            current_chunk += sent + " "
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append({
            "title": section["title"],
            "content": current_chunk.strip()
        })

    return chunks

def further_chunk_sections(sections, max_tokens):

    """
    Apply token-based chunking to all sections.
    """

    final_chunks = []

    for sec in sections:
        sub_chunks = chunk_section_by_tokens(sec, max_tokens=max_tokens)
        final_chunks.extend(sub_chunks)

    return final_chunks



def save_to_json(data, filename):

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {filename}")

# Move the FAISS indexing function inside the endpoint (or import it there)
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Ensure 'uploads' directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Save the uploaded PDF to disk
    file_path = os.path.join("uploads", file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Convert PDF to Markdown using pymupdf4llm
    md_text = pymupdf4llm.to_markdown(file_path)

    # Process the markdown text directly (using a function that processes text rather than a file path)
    sections = preprocess_markdown_file(md_text)
    
    # Save the processed sections to a JSON file
    json_filename = os.path.join("uploads", f"{os.path.splitext(file.filename)[0]}_processed.json")
    save_to_json(sections, json_filename)
    
    # Derive the title automatically from the file name (without extension)
    title = os.path.splitext(file.filename)[0]
    
    # Create a new Contract record in the database
    from models import Contract  # ensure Contract is imported if not globally
    contract = Contract(
        title=title,
        file_path=file_path,
        json_path=json_filename,
    )
    db.add(contract)
    db.commit()
    db.refresh(contract)
    
    return {
        "id": contract.id,
        "title": contract.title,
        "file_path": contract.file_path,
        "json_path": contract.json_path,
    }


# Other endpoints remain unchanged...
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/contracts")
def list_contracts(db: Session = Depends(get_db)):
    from models import Contract  # ensure Contract is imported
    return db.query(Contract).all()

from fastapi import HTTPException

# ----------  GET Contrat by ID ---------

@app.get("/contracts/{contract_id}")
def get_contract(contract_id: int, db: Session = Depends(get_db)):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return contract



from fastapi import HTTPException
@app.delete("/contracts")
def delete_all_contracts(db: Session = Depends(get_db)):
    contracts = db.query(Contract).all()
    if not contracts:
        raise HTTPException(status_code=404, detail="No contracts found")
    for contract in contracts:
        db.delete(contract)
    db.commit()
    return {"message": "All contracts deleted"}



@app.post("/retrieve-context")
def retrieve_context_endpoint(question: str = Body(...), contract_id: int = Body(...), db: Session = Depends(get_db), top_k=3):
    
    from models import Contract
    # Lookup contract to get paths (if needed)
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    # Load JSON and FAISS index
    try:
        with open(contract.json_path, 'r', encoding='utf-8') as f:
            sections = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON file: {str(e)}")
    

    chunks = further_chunk_sections(sections, max_tokens=400)
    

    retriever = FaissRetriever()

    retriever.build_index(chunks)

    query = question

    retrieved = retriever.search(query, top_k=top_k)

    context = " ".join([item["text"] for item in retrieved])

    return {"context": context}


from fastapi import HTTPException
from pydantic import BaseModel

class ConversationResponse(BaseModel):
    contract_id: int
    messages: list

@app.get("/conversations/{contract_id}", response_model=ConversationResponse)
def get_conversation(contract_id: int, db: Session = Depends(get_db)):
    from models import Conversation
    conversation = db.query(Conversation).filter(Conversation.contract_id == contract_id).first()
    if not conversation:
        # If no conversation exists, you might choose to return an empty conversation:
        return {"contract_id": contract_id, "messages": []}
    # Assuming messages are stored as a JSON string, parse them.
    try:
        messages = json.loads(conversation.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error parsing conversation messages")
    return {"contract_id": contract_id, "messages": messages}



class ConversationRequest(BaseModel):
    contract_id: int
    messages: list  # List of message objects, e.g., [{ "sender": "user", "text": "Hello" }, ...]

@app.post("/conversations", response_model=ConversationResponse)
def save_conversation(conversation_request: ConversationRequest, db: Session = Depends(get_db)):
    from models import Conversation, Contract
    # Optionally, you can validate that the contract exists.
    contract = db.query(Contract).filter(Contract.id == conversation_request.contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Try to retrieve an existing conversation
    conversation = db.query(Conversation).filter(Conversation.contract_id == conversation_request.contract_id).first()
    messages_json = json.dumps(conversation_request.messages)
    if conversation:
        conversation.messages = messages_json
    else:
        conversation = Conversation(
            contract_id=conversation_request.contract_id,
            messages=messages_json
        )
        db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return {"contract_id": conversation.contract_id, "messages": conversation_request.messages}



@app.delete("/contracts/{contract_id}")
def delete_contract(contract_id: int, db: Session = Depends(get_db)):
    from models import Contract  # Ensure Contract is imported
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    db.delete(contract)
    db.commit()
    return {"message": "Contract deleted"}

@app.delete("/conversations/{contract_id}")
def delete_conversation(contract_id: int, db: Session = Depends(get_db)):
    from models import Conversation  # Ensure Conversation is imported
    conversation = db.query(Conversation).filter(Conversation.contract_id == contract_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted"}
