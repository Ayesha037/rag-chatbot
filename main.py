from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List
import asyncio
import warnings
import logging
import traceback
import shutil
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = FastAPI(
    title="RAG Document Intelligence System",
    description="Production RAG pipeline: PDF upload → FAISS vector search → LLaMA3 answer generation via Groq",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
qa_chain = None
uploaded_docs = []
vectorstore_path = "vectorstore/faiss_index"

Path("vectorstore").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class UploadResponse(BaseModel):
    message: str
    filename: str
    size_bytes: int
    chunks: int

logger.info("🚀 Initializing RAG system...")

try:
    logger.info("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Embeddings model loaded")
except Exception as e:
    logger.error(f"❌ Error loading embeddings: {str(e)}")
    embeddings = None

try:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("❌ GROQ_API_KEY not found in .env file!")
        llm = None
    else:
        logger.info("Loading LLM (Groq)...")
        llm = ChatGroq(
            temperature=0,
            model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_key,
            max_tokens=1024
        )
        logger.info("✅ LLM loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading LLM: {str(e)}")
    llm = None

def load_vectorstore():
    """Load existing vectorstore"""
    global vector_store
    try:
        if os.path.exists(vectorstore_path):
            logger.info(f"Loading existing vectorstore from {vectorstore_path}...")
            vector_store = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vectorstore loaded")
            return True
    except Exception as e:
        logger.warning(f"Could not load vectorstore: {str(e)}")
    return False

def setup_qa_chain():
    """Setup QA chain"""
    global qa_chain, vector_store
    
    if vector_store is None or llm is None:
        logger.warning("Cannot setup QA chain - missing vectorstore or llm")
        return False
    
    try:
        logger.info("Setting up QA chain...")
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 3))}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logger.info("✅ QA chain setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        return False

load_vectorstore()
if vector_store:
    setup_qa_chain()

logger.info("✅ RAG system initialized")


@app.get("/")
def root():
    """Root endpoint - API info"""
    return {
        "app": "RAG Document Intelligence System",
        "version": "1.0.0",
        "author": "Mohammad Ayesha Summaiyya",
        "github": "https://github.com/Ayesha037",
        "status": "running",
        "endpoints": {
            "/upload": "POST - Upload PDF files",
            "/query": "POST - Ask questions about uploaded documents",
            "/health": "GET - Check system health",
            "/documents": "GET - List uploaded documents",
            "/reset": "GET - Reset system",
            "/docs": "GET - API documentation (Swagger)"
        }
    }

@app.get("/health")
def health():
    """Check system health"""
    return {
        "status": "running",
        "models_loaded": embeddings is not None and llm is not None,
        "vector_store": "initialized" if vector_store else "not_initialized",
        "qa_chain": "ready" if qa_chain else "not_ready",
        "documents_uploaded": len(uploaded_docs),
        "api_url": "http://localhost:7860"
    }

@app.get("/documents")
def list_documents():
    """List all uploaded documents"""
    return {
        "count": len(uploaded_docs),
        "documents": uploaded_docs
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    global vector_store, qa_chain
    
    logger.info(f"📤 Uploading: {file.filename}")

    if not file.filename.endswith(".pdf"):
        logger.error(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files accepted. Got: {file.filename}"
        )
    
    if embeddings is None or llm is None:
        logger.error("Models not loaded!")
        raise HTTPException(
            status_code=500,
            detail="Models not initialized. Check GROQ_API_KEY in .env file"
        )
    
    tmp_path = None
    try:
        
        logger.info("Reading file...")
        contents = await file.read()
        file_size_bytes = len(contents)
        logger.info(f"File read: {file_size_bytes} bytes")
        
        logger.info("Writing to temporary file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(contents)
            tmp_file.flush()  
            tmp_path = tmp_file.name
        
        logger.info(f"Temp file created at: {tmp_path}")

        logger.info(f"Loading PDF: {file.filename}")
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            num_pages = len(documents)
        except Exception as pdf_error:
            logger.error(f"PDF Loading error: {str(pdf_error)}")
            raise Exception(f"Failed to load PDF: {str(pdf_error)}")
        
        if num_pages == 0:
            raise Exception("PDF has no pages or is corrupted!")
        
        logger.info(f"PDF loaded: {num_pages} pages")
        
    
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
        chunks = text_splitter.split_documents(documents)
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            raise Exception("No chunks created from PDF!")
        
        logger.info(f"Created {num_chunks} chunks")
        
        logger.info("Creating/updating vectorstore...")
        try:
            if vector_store is None:
                logger.info("Creating new vectorstore...")
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                logger.info("Adding to existing vectorstore...")
                vector_store.add_documents(chunks)
        except Exception as faiss_error:
            logger.error(f"FAISS error: {str(faiss_error)}")
            raise Exception(f"Failed to create vectorstore: {str(faiss_error)}")
        
        logger.info(f"Saving vectorstore to {vectorstore_path}...")
        try:
 
            Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(vectorstore_path)
            logger.info("Vectorstore saved successfully")
        except Exception as save_error:
            logger.error(f"Error saving vectorstore: {str(save_error)}")
            logger.warning("Continuing without saving (vectorstore in memory)")
        
        if not setup_qa_chain():
            logger.warning("QA chain setup failed, but continuing")
        
        uploaded_docs.append({
            "filename": file.filename,
            "size_bytes": file_size_bytes,
            "chunks": num_chunks,
            "pages": num_pages
        })
        
        logger.info(f"✅ Successfully processed {file.filename}")
        
        return {
            "message": f"Successfully uploaded and indexed '{file.filename}'",
            "filename": file.filename,
            "size_bytes": file_size_bytes,
            "chunks": num_chunks,
            "pages": num_pages,
            "total_documents": len(uploaded_docs)
        }
    
    except Exception as e:
        logger.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Temp file cleaned up: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {str(cleanup_error)}")

@app.post("/query")
async def query(request: QueryRequest):
    """Ask a question about uploaded documents"""
    global qa_chain
    
    logger.info(f"❓ Query: {request.question}")
    
    if qa_chain is None or vector_store is None:
        logger.error("No documents uploaded yet")
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a PDF first."
        )
    
    try:
        logger.info("Processing query...")
        result = qa_chain({"query": request.question})
        
        answer = result.get("result", "No answer generated")
        source_docs = result.get("source_documents", [])
        
        logger.info(f"Got answer from {len(source_docs)} sources")
        
        sources = []
        for doc in source_docs[:request.top_k]:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "unknown")
            })
        
        logger.info("✅ Query processed successfully")
        
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "model": "Groq LLaMA3",
            "confidence": "high" if sources else "low"
        }
    
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/reset")
def reset_system():
    """Reset system - clear all documents"""
    global vector_store, qa_chain, uploaded_docs
    
    logger.info("🔄 Resetting system...")
    
    try:
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            logger.info("Vectorstore cleared")
        
        vector_store = None
        qa_chain = None
        uploaded_docs = []
        
        logger.info("✅ System reset complete")
        
        return {
            "status": "reset",
            "message": "System reset complete. Ready for new documents."
        }
    
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting system: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Starting RAG Document Intelligence System")
    logger.info("=" * 60)
    logger.info("API URL: http://localhost:7860")
    logger.info("API Docs: http://localhost:7860/docs")
    logger.info("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=7860)