# RAG Document Intelligence System

A production-grade document Q&A assistant — upload any PDF, ask questions in plain English, get source-attributed answers.

Powered by **LangChain · LLaMA3 (Groq) · FAISS · Streamlit · FastAPI**.

---

## Architecture

```
PDF Upload → Text Chunking → HuggingFace Embeddings
         → FAISS Vector Store → LLaMA3 (Groq) → Source-attributed Answer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA3-8b via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Orchestration | LangChain |
| PDF Parsing | PyMuPDF |
| Backend API | FastAPI (port 7860) |
| Frontend | Streamlit (port 8501) |

---

## Quick Start (local)

```bash
git clone https://github.com/Ayesha037/rag-chatbot
cd rag-chatbot

pip install -r requirements.txt

# Copy and fill in your Groq API key
cp .env.example .env
# Edit .env → set GROQ_API_KEY=gsk_...

# Terminal 1 — backend
python main.py

# Terminal 2 — frontend
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Docker

```bash
docker build -t rag-chatbot .
docker run --env-file .env -p 7860:7860 -p 8501:8501 rag-chatbot
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload & index a PDF |
| POST | `/query` | Ask a question |
| GET | `/documents` | List indexed docs |
| GET | `/health` | Health check |
| GET | `/reset` | Clear all docs |
| GET | `/docs` | Swagger UI |

---

## Key Fixes (v2.0)

- **Switched `PyPDFLoader` → `PyMuPDFLoader`** — no extra `pypdf` dependency needed; more reliable PDF parsing
- **Proper error messages** returned to frontend (no more "Unknown error")
- **CORS** correctly configured for Streamlit ↔ FastAPI communication
- **Frontend** shows real error text, source excerpts, page numbers, confidence indicator
- **Docker** runs both services in one container

---

*Built by [Mohammad Ayesha Summaiyya](https://github.com/Ayesha037)*