import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

from services.pdf_processor import process_pdf
from services.chunker import DocumentChunker
from services.embeddings import EmbeddingManager
from services.rag_pipeline import RAGPipeline
from utils.helpers import (
    save_uploaded_file,
    get_uploaded_files,
    clear_data_directory,
    clear_vectorstore,
    format_file_size,
    ensure_directories
)

load_dotenv()

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload MULTIPLE PDFs and ask questions!",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
embedding_manager: Optional[EmbeddingManager] = None
rag_pipeline: Optional[RAGPipeline] = None
chunker = DocumentChunker()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_history: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    citations: list
    question: str
    model: str
    context_used: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int
    file_size: str
    total_documents: int


class ResetResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    pipeline_ready: bool
    uploaded_files: list
    total_files: int


def get_or_create_pipeline() -> RAGPipeline:
    global embedding_manager, rag_pipeline

    if rag_pipeline is not None:
        return rag_pipeline

    try:
        embedding_manager = EmbeddingManager()
        embedding_manager.load_vectorstore()
        rag_pipeline = RAGPipeline(
            vectorstore=embedding_manager.get_vectorstore()
        )
        logger.info("Pipeline loaded from existing vectorstore")
        return rag_pipeline

    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(
                "No documents uploaded yet. "
                "Please upload a PDF first!"
            )
        )


@app.on_event("startup")
async def startup_event():
    """Pre-load pipeline on server startup."""
    global embedding_manager, rag_pipeline
    ensure_directories()
    logger.info("Server starting up...")

    try:
        manager = EmbeddingManager()
        if manager.vectorstore_exists():
            logger.info("Found existing vectorstore, pre-loading...")
            manager.load_vectorstore()
            embedding_manager = manager
            rag_pipeline = RAGPipeline(
                vectorstore=embedding_manager.get_vectorstore()
            )
            files = get_uploaded_files()
            logger.success(
                f"Pipeline pre-loaded with "
                f"{len(files)} document(s)!"
            )
        else:
            logger.info("No vectorstore found, waiting for upload...")
    except Exception as e:
        logger.warning(f"Pre-load skipped: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ensure_directories()
    vs_ready = False

    if embedding_manager is not None:
        vs_ready = embedding_manager.get_vectorstore() is not None
    else:
        vs_path = os.getenv(
            "VECTORSTORE_PATH", "vectorstore/faiss_index"
        )
        vs_ready = (
            Path(vs_path + ".faiss").exists() or
            Path(vs_path).is_dir()
        )

    files = get_uploaded_files()

    return HealthResponse(
        status="healthy",
        vectorstore_ready=vs_ready,
        pipeline_ready=rag_pipeline is not None,
        uploaded_files=[Path(f).name for f in files],
        total_files=len(files)
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    Supports multiple PDFs — each new PDF is ADDED
    to the existing vectorstore, not replacing it.
    Ask questions across ALL uploaded documents!
    """
    global embedding_manager, rag_pipeline

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported!"
        )

    logger.info(f"New upload: {file.filename}")

    try:
        file_content = await file.read()
        file_size = len(file_content)
        saved_path = save_uploaded_file(
            file_content, file.filename
        )
        logger.info(f"Saved: {saved_path}")

        logger.info("Extracting text...")
        processed_doc = process_pdf(saved_path)
        logger.info(
            f"Extracted: {processed_doc.total_pages} pages, "
            f"{len(processed_doc.full_text):,} chars"
        )

        logger.info("Chunking...")
        chunks = chunker.chunk_document(processed_doc)
        langchain_docs = chunker.chunks_to_langchain_docs(chunks)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Adding to vectorstore...")
        if embedding_manager is None:
            embedding_manager = EmbeddingManager()

        embedding_manager.add_documents(langchain_docs)
        embedding_manager.save_vectorstore()
        logger.info("Vectorstore updated and saved!")

        rag_pipeline = RAGPipeline(
            vectorstore=embedding_manager.get_vectorstore()
        )

        total_docs = len(get_uploaded_files())

        logger.success(
            f" '{file.filename}' added! "
            f"Total documents: {total_docs}"
        )

        return UploadResponse(
            message=(
                f"'{file.filename}' added successfully! "
                f"Total documents: {total_docs}"
            ),
            filename=file.filename,
            pages=processed_doc.total_pages,
            chunks=len(chunks),
            file_size=format_file_size(file_size),
            total_documents=total_docs
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question.
    Searches across ALL uploaded documents
    and returns the most relevant answer.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty!"
        )

    logger.info(f"Query: '{request.question[:60]}'")

    try:
        pipeline = get_or_create_pipeline()

        if request.top_k:
            pipeline.retriever.top_k = request.top_k

        result = pipeline.query(
            question=request.question,
            use_history=request.use_history
        )

        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            question=result["question"],
            model=result["model"],
            context_used=result["context"][:500]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/reset", response_model=ResetResponse)
async def reset_system():
    """
    Reset everything.
    Clears ALL uploaded PDFs and vectorstore.
    Use this to start fresh with new documents.
    """
    global embedding_manager, rag_pipeline

    try:
        if rag_pipeline is not None:
            rag_pipeline.clear_history()

        embedding_manager = None
        rag_pipeline = None

        clear_data_directory()
        clear_vectorstore()

        logger.info("System reset complete")

        return ResetResponse(
            message=(
                "Reset successful! "
                "All documents cleared. "
                "Upload new PDFs to get started."
            )
        )

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )


@app.get("/files")
async def list_files():
    """List all uploaded PDF files."""
    files = get_uploaded_files()
    return {
        "files": [Path(f).name for f in files],
        "count": len(files)
    }


@app.get("/history")
async def get_chat_history():
    """Get current chat history."""
    if rag_pipeline is None:
        return {
            "history": [],
            "message": "No active session"
        }
    return {
        "history": rag_pipeline.get_history(),
        "count": len(rag_pipeline.get_history())
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting RAG API on port {port}...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )