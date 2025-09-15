import os
import re
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_utils import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from mistral_client import ask_mistral

app = FastAPI()
vector_store = None
memory = {}  # Global memory for personal info

class AskRequest(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Upload PDF -------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global vector_store
    # file_path = f"temp_{file.filename}"
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    return {"status": "success", "chunks": len(chunks), "filename": file.filename}

@app.post("/ask/")
async def ask_question(request: AskRequest):
    global vector_store, memory
    query = request.query.strip()

    # ----------------- Update memory with personal info -----------------
    match_name = re.search(r"my name is (\w+ \w+|\w+)", query, re.IGNORECASE)
    if match_name:
        memory["name"] = match_name.group(1)

    match_age = re.search(r"i am (\d+) years? old", query, re.IGNORECASE)
    if match_age:
        memory["age"] = int(match_age.group(1))

    # ----------------- PDF Context -----------------
    if vector_store and hasattr(vector_store, "get_relevant_chunks"):
        relevant_chunks = vector_store.get_relevant_chunks(query)
        context_text = "\n".join(relevant_chunks) if relevant_chunks else ""
    else:
        context_text = None  # No PDF uploaded

    # ----------------- Memory Context -----------------
    memory_text = "\n".join([f"{k}: {v}" for k, v in memory.items()]) if memory else "No memory available."

    # ----------------- Construct Prompt -----------------
    if context_text:  # PDF exists
        prompt = f"""
You are a helpful AI assistant.
User asked: {query}

Memory:
{memory_text}

Refer to PDF:
{context_text}

Instructions:
- Answer concisely using memory if relevant.
- Use PDF content if relevant.
- Otherwise answer using general knowledge.
- Update memory if user provides new personal info.
- Keep answers short and to the point.
"""
    else:  # No PDF uploaded
        prompt = f"""
You are a helpful AI assistant.
User asked: {query}

Memory:
{memory_text}

Instructions:
- Answer concisely using memory if relevant.
- Otherwise answer using general knowledge.
- Keep answers short and to the point.
"""

    # ----------------- Get AI Response -----------------
    response_text = ask_mistral(prompt, query)

    return {"answer": response_text, "memory": memory, "pdf_uploaded": bool(context_text)}

