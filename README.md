# 📄 RAG Document Intelligence System

> Upload any PDF → ask questions in plain English → get accurate, source-attributed answers instantly.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://rag-chatbot-danjpg2yc4taeeerwzsvyp.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Ayesha037-181717?style=for-the-badge&logo=github)](https://github.com/Ayesha037/rag-chatbot)

---

## 🌐 Live Demo

👉 **[https://rag-chatbot-danjpg2yc4taeeerwzsvyp.streamlit.app/](https://rag-chatbot-danjpg2yc4taeeerwzsvyp.streamlit.app/)**

---

## 🚀 What it does

1. Upload any PDF document
2. Ask questions in plain English
3. Get accurate answers with source page references — no hallucinations

---

## ⚙️ Architecture

```
PDF Upload → Text Chunking → HuggingFace Embeddings
          → FAISS Vector Store → LLaMA3 (Groq) → Source-attributed Answer
```

---

## 🛠 Tech Stack

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

## 📊 Performance

- ⚡ Sub-10s response latency
- 💰 $0 API cost — fully open source stack
- 🎯 100% source attribution — eliminates hallucinations
- 🔧 Configurable top-k retrieval (1–10 sources)

---

## 🔧 Run Locally

```bash
git clone https://github.com/Ayesha037/rag-chatbot
cd rag-chatbot

pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env
# Edit .env → set GROQ_API_KEY=gsk_...

# Terminal 1 — backend
python main.py

# Terminal 2 — frontend
streamlit run frontend/app.py
```

Open **http://localhost:8501**

---

## 🐳 Docker

```bash
docker build -t rag-chatbot .
docker run --env-file .env -p 7860:7860 -p 8501:8501 rag-chatbot
```

---

## 📄 API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload & index a PDF |
| POST | `/query` | Ask a question |
| GET | `/documents` | List indexed docs |
| GET | `/health` | Health check |
| GET | `/reset` | Clear all docs |
| GET | `/docs` | Swagger UI |

---

## 📁 Project Structure

```
rag-chatbot/
├── main.py              # FastAPI backend
├── frontend/
│   └── app.py           # Streamlit frontend (all-in-one for cloud deploy)
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

*Built by [Mohammad Ayesha Summaiyya](https://github.com/Ayesha037)*
