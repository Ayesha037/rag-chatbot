from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Document Intelligence System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = None
qa_chain = None
uploaded_docs = []

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

logger.info("🚀 Initializing RAG system...")

# Load embeddings
try:
    logger.info("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Embeddings loaded")
except Exception as e:
    logger.error(f"❌ Embeddings error: {e}")
    embeddings = None

# Load LLM
try:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("❌ GROQ_API_KEY missing!")
        llm = None
    else:
        logger.info("Loading Groq LLM...")
        llm = ChatGroq(
            temperature=0,
            model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_key,
            max_tokens=1024
        )
        logger.info("✅ LLM loaded")
except Exception as e:
    logger.error(f"❌ LLM error: {e}")
    llm = None

def setup_qa_chain():
    """Setup QA chain"""
    global qa_chain, vector_store
    
    if not vector_store or not llm:
        return False
    
    try:
        prompt = PromptTemplate(
            template="""Use context to answer the question.
            
Context: {context}
Question: {question}
Answer:""",
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logger.info("✅ QA chain ready")
        return True
    except Exception as e:
        logger.error(f"❌ QA chain error: {e}")
        return False

logger.info("✅ RAG initialized")

@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "running",
        "models_loaded": embeddings is not None and llm is not None,
        "documents_ready": vector_store is not None
    }

@app.get("/documents")
def list_documents():
    return {"count": len(uploaded_docs), "documents": uploaded_docs}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF"""
    global vector_store, qa_chain
    
    logger.info(f"Uploading: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files")
    
    if not embeddings or not llm:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    tmp_path = None
    try:
        # Read file
        logger.info("Reading file...")
        contents = await file.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name
        
        logger.info(f"Processing PDF with PyMuPDF...")
        
        # Load PDF using PyMuPDFLoader (uses pymupdf which is in requirements)
        try:
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
        except Exception as pdf_error:
            logger.error(f"PDF Loading error: {pdf_error}")
            raise Exception(f"Failed to load PDF: {str(pdf_error)}")
        
        if not documents:
            raise Exception("PDF is empty")
        
        logger.info(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            raise Exception("No chunks created")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create FAISS index (in memory only)
        logger.info("Creating embeddings...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        logger.info("Setting up QA chain...")
        setup_qa_chain()
        
        # Record document
        uploaded_docs.append({
            "filename": file.filename,
            "size": len(contents),
            "chunks": len(chunks),
            "pages": len(documents)
        })
        
        logger.info(f"✅ Success: {file.filename}")
        
        return {
            "message": "Success",
            "filename": file.filename,
            "size_bytes": len(contents),
            "pages": len(documents),
            "chunks": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except:
                pass

@app.post("/query")
async def query(request: QueryRequest):
    """Ask question"""
    if not qa_chain:
        raise HTTPException(status_code=400, detail="Upload PDF first")
    
    try:
        result = qa_chain({"query": request.question})
        
        sources = []
        for doc in result.get("source_documents", [])[:3]:
            sources.append({
                "content": doc.page_content[:100],
                "page": doc.metadata.get("page", 0)
            })
        
        return {
            "question": request.question,
            "answer": result.get("result", "No answer"),
            "sources": sources,
            "model": "Groq LLaMA3"
        }
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reset")
def reset():
    global vector_store, qa_chain, uploaded_docs
    vector_store = None
    qa_chain = None
    uploaded_docs = []
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)