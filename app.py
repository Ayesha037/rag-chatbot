from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI(
    title="RAG Document Intelligence System",
    description="Production RAG pipeline: PDF upload → FAISS vector search → LLaMA3 answer generation via Groq",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {
        "app": "RAG Document Intelligence System",
        "author": "Mohammad Ayesha Summaiyya",
        "github": "https://github.com/Ayesha037",
        "endpoints": ["/upload", "/query", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files accepted"})
    contents = await file.read()
    return {
        "message": f"Received '{file.filename}' ({len(contents)} bytes). Connect your RAG pipeline here.",
        "filename": file.filename,
        "size_bytes": len(contents)
    }

@app.post("/query")
async def query(request: QueryRequest):
    return {
        "question": request.question,
        "answer": "Connect your LangChain + Groq pipeline here to generate answers.",
        "top_k": request.top_k,
        "sources": ["doc_chunk_1", "doc_chunk_2", "doc_chunk_3"]
    }