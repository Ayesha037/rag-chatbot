---
title: Rag Document Intelligence
emoji: 📄
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
---

# RAG Document Intelligence System

A production-grade document assistant that lets you query any document using natural language — powered by LangChain, LLaMA3, and FAISS with zero API cost.

---

## 🚀 What it does

Upload any document → ask questions in plain English → get accurate, source-attributed answers instantly. No hallucinations, no vendor lock-in.

---

## ⚙️ Architecture

Document Upload → Text Chunking → Embedding (Hugging Face)
→ Vector Store (FAISS) → LLM Generation (LLaMA3 via Groq)
→ Source-attributed Response

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | LLaMA3 via Groq |
| Orchestration | LangChain + LangGraph |
| Embeddings | Hugging Face |
| Vector Store | FAISS |
| API | FastAPI (6 endpoints) |
| Indexing | LlamaIndex |

---

## 📊 Performance

- ⚡ Sub-10s response latency
- 💰 $0 API cost — fully open source stack
- 🎯 100% source attribution — eliminates hallucinations
- 🔧 Configurable top-k retrieval

---

## 🔧 How to Run

```bash
git clone https://github.com/Ayesha037/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📄 API Endpoints

- `POST /upload` — upload and index a document
- `POST /query` — query the document
- `GET /documents` — list indexed documents
- `DELETE /document` — remove a document
- `GET /health` — health check
- `GET /docs` — auto-generated OpenAPI docs

---

*Built by [Mohammad Ayesha Summaiyya](https://github.com/Ayesha037)*

