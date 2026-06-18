from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# App init
app = FastAPI(title="RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
vector_store = None
qa_chain = None
uploaded_docs = []

# Request model
class QueryRequest(BaseModel):
    question: str

# ---------------------------
# LOAD MODELS
# ---------------------------
try:
    logger.info("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embeddings loaded")
except Exception as e:
    logger.error(f"Embedding error: {e}")
    embeddings = None

try:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("Missing GROQ_API_KEY")

    logger.info("Loading LLM...")
    llm = ChatGroq(
        api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    logger.info("LLM loaded")
except Exception as e:
    logger.error(f"LLM error: {e}")
    llm = None


# ---------------------------
# QA SETUP
# ---------------------------
def setup_chain():
    global qa_chain

    if not vector_store or not llm:
        return False

    prompt = PromptTemplate(
        template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return True


# ---------------------------
# ROUTES
# ---------------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {
        "models_loaded": embeddings is not None and llm is not None,
        "documents_loaded": vector_store is not None
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vector_store, uploaded_docs

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    try:
        # Save temp file
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"Processing: {file.filename}")

        # Load PDF (uses pymupdf)
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise Exception("Empty PDF")

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # Vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Setup QA
        setup_chain()

        uploaded_docs.append(file.filename)

        return {
            "message": "Uploaded successfully",
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

    finally:
        if "tmp_path" in locals() and Path(tmp_path).exists():
            Path(tmp_path).unlink()


@app.post("/query")
async def query(req: QueryRequest):
    if not qa_chain:
        raise HTTPException(400, "Upload PDF first")

    try:
        result = qa_chain({"query": req.question})

        return {
            "answer": result["result"],
            "sources": [
                doc.page_content[:150]
                for doc in result["source_documents"]
            ]
        }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, str(e))


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