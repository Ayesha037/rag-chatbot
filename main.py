from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import logging
import traceback
import tempfile
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyMuPDFLoader   # uses pymupdf, no extra dep
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Paths ──────────────────────────────────────────────────────────────────────
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
Path("vectorstore").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Intelligence System",
    description="PDF upload → FAISS vector search → LLaMA3 answer generation via Groq",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
vector_store = None
qa_chain = None
uploaded_docs: List[dict] = []

# ── Pydantic models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# ── Model initialisation ───────────────────────────────────────────────────────
logger.info("🚀 Initialising RAG system…")

try:
    logger.info("Loading embeddings model (sentence-transformers/all-MiniLM-L6-v2)…")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    logger.info("✅ Embeddings model loaded")
except Exception as e:
    logger.error(f"❌ Embeddings failed: {e}")
    embeddings = None

try:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise ValueError("GROQ_API_KEY is empty or missing in .env")
    logger.info("Loading LLM (Groq)…")
    llm = ChatGroq(
        temperature=0,
        model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=groq_key,
        max_tokens=1024,
    )
    logger.info("✅ LLM loaded")
except Exception as e:
    logger.error(f"❌ LLM failed: {e}")
    llm = None

# ── Helpers ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If the answer is not contained in the context, say "I don't know" — do NOT make up an answer.

Context:
{context}

Question: {question}

Answer:"""


def _build_qa_chain() -> bool:
    global qa_chain
    if vector_store is None or llm is None:
        return False
    try:
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 3))}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        logger.info("✅ QA chain ready")
        return True
    except Exception as e:
        logger.error(f"QA chain error: {e}")
        return False


def _load_vectorstore() -> bool:
    global vector_store
    try:
        if embeddings and os.path.exists(VECTORSTORE_PATH):
            logger.info(f"Loading vectorstore from {VECTORSTORE_PATH}…")
            vector_store = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("✅ Vectorstore loaded")
            return True
    except Exception as e:
        logger.warning(f"Could not load vectorstore: {e}")
    return False


# Load on startup
_load_vectorstore()
if vector_store:
    _build_qa_chain()
logger.info("✅ Startup complete")

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "RAG Document Intelligence System",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /upload": "Upload a PDF file",
            "POST /query": "Ask a question about uploaded documents",
            "GET  /health": "System health",
            "GET  /documents": "List uploaded documents",
            "GET  /reset": "Clear all documents",
            "GET  /docs": "Swagger UI",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "embeddings_loaded": embeddings is not None,
        "llm_loaded": llm is not None,
        "vector_store": "ready" if vector_store else "empty",
        "qa_chain": "ready" if qa_chain else "not_ready",
        "documents_uploaded": len(uploaded_docs),
    }


@app.get("/documents")
def list_documents():
    return {"count": len(uploaded_docs), "documents": uploaded_docs}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    global vector_store, qa_chain

    # ── Validations ────────────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, f"Only PDF files are accepted. Got: {file.filename}")

    if embeddings is None:
        raise HTTPException(500, "Embeddings model not initialised. Check server logs.")

    if llm is None:
        raise HTTPException(
            500,
            "LLM not initialised. Ensure GROQ_API_KEY is set correctly in .env"
        )

    tmp_path = None
    try:
        # Save upload to a temp file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(400, "Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"📄 Processing '{file.filename}' ({len(contents)} bytes)…")

        # Load PDF with PyMuPDF (no extra pypdf dep needed)
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(400, "PDF appears to be empty or unreadable.")

        num_pages = len(documents)
        logger.info(f"  Loaded {num_pages} page(s)")

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise HTTPException(400, "Could not extract any text from this PDF.")

        num_chunks = len(chunks)
        logger.info(f"  Created {num_chunks} chunk(s)")

        # Build / update vectorstore
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("  Created new vectorstore")
        else:
            vector_store.add_documents(chunks)
            logger.info("  Updated existing vectorstore")

        vector_store.save_local(VECTORSTORE_PATH)
        _build_qa_chain()

        uploaded_docs.append({
            "filename": file.filename,
            "size_bytes": len(contents),
            "pages": num_pages,
            "chunks": num_chunks,
        })

        logger.info(f"✅ '{file.filename}' indexed successfully")
        return {
            "message": f"'{file.filename}' uploaded and indexed successfully.",
            "filename": file.filename,
            "size_bytes": len(contents),
            "pages": num_pages,
            "chunks": num_chunks,
            "total_documents": len(uploaded_docs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Error processing file: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query")
async def query(request: QueryRequest):
    """Ask a question about the uploaded documents."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if qa_chain is None or vector_store is None:
        raise HTTPException(
            400,
            "No documents have been indexed yet. Please upload a PDF first."
        )

    try:
        logger.info(f"❓ Query: {request.question}")
        result = qa_chain({"query": request.question})

        answer = result.get("result", "No answer generated.")
        source_docs = result.get("source_documents", [])

        sources = [
            {
                "content": doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""),
                "page": doc.metadata.get("page", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in source_docs[: request.top_k]
        ]

        logger.info(f"✅ Answer generated from {len(sources)} source(s)")
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            "confidence": "high" if sources else "low",
        }

    except Exception as e:
        logger.error(f"❌ Query error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Error processing query: {str(e)}")


@app.get("/reset")
def reset_system():
    """Clear all indexed documents and reset the system."""
    global vector_store, qa_chain, uploaded_docs

    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)

        vector_store = None
        qa_chain = None
        uploaded_docs = []

        logger.info("🔄 System reset complete")
        return {"status": "reset", "message": "All documents cleared. Ready for new uploads."}

    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(500, f"Reset failed: {str(e)}")


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exc_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀  RAG Document Intelligence System  v2.0")
    logger.info("=" * 60)
    logger.info("Backend  →  http://localhost:7860")
    logger.info("Swagger  →  http://localhost:7860/docs")
    logger.info("Frontend →  http://localhost:8501  (run app.py separately)")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)