import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf_utils import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from mistral_client import ask_mistral
from pydantic import BaseModel

app = FastAPI()
vector_store = None
user_data = {}
class AskRequest(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global vector_store
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    return {"status": "success", "chunks": len(chunks)}

@app.post("/ask/")
async def ask_question(request: AskRequest):
    query = request.query
    global vector_store

    context_text = ""

    # If PDF exists, retrieve relevant chunks
    if vector_store and hasattr(vector_store, "get_relevant_chunks"):
        relevant_chunks = vector_store.get_relevant_chunks(query)
        if relevant_chunks:
            context_text = "\n".join(relevant_chunks)

    # Construct prompt for Mistral
    if context_text:
        prompt = f"""
You are a helpful AI assistant.
Refer to this excerpt from the uploaded PDF:
{context_text}

User asked: {query}
Answer based on the PDF content if possible.
If not in PDF, answer based on your general knowledge.
Be clear, professional, and detailed.
"""
    else:
        prompt = f"""
You are a helpful AI assistant.
User asked: {query}
Answer based on general knowledge.
Be clear, professional, and detailed.
"""

    # Call Mistral
    response = ask_mistral(prompt, query)  # pass both prompt and query

    return {"answer": response}



@app.post("/save_user/")
async def save_user(name: str = Form(None), email: str = Form(None)):
    if name:
        user_data["name"] = name
    if email:
        user_data["email"] = email
    return {"status": "saved", "user_data": user_data}

@app.get("/get_user/")
async def get_user():
    return user_data
