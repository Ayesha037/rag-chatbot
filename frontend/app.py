"""
RAG Document Intelligence — All-in-one Streamlit app
Starts FastAPI backend in a background thread, then runs Streamlit UI.
Works on Streamlit Cloud, Railway, HuggingFace Spaces, or locally.
"""

import threading
import time
import os
import tempfile
import shutil
import warnings
import logging
import traceback
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Load env ───────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── RAG globals ────────────────────────────────────────────────────────────────
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
Path("vectorstore").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ── Lazy-init models (shared across Streamlit reruns via st.cache_resource) ───
import streamlit as st

@st.cache_resource(show_spinner="Loading AI models… (first run only, ~30s)")
def load_models():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found. Add it in Streamlit Secrets or .env")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    llm = ChatGroq(
        temperature=0,
        model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=groq_key,
        max_tokens=1024,
    )
    return embeddings, llm


@st.cache_resource(show_spinner=False)
def load_vectorstore(_embeddings):
    from langchain_community.vectorstores import FAISS
    try:
        if os.path.exists(VECTORSTORE_PATH):
            vs = FAISS.load_local(
                VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True
            )
            logger.info("✅ Vectorstore loaded from disk")
            return vs
    except Exception as e:
        logger.warning(f"Could not load vectorstore: {e}")
    return None


def build_qa_chain(llm, vector_store):
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        template="""Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 3))}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def process_pdf(file_bytes, filename, embeddings):
    """Process uploaded PDF and return (vector_store, num_pages, num_chunks)."""
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        if not documents:
            raise ValueError("PDF appears empty or unreadable.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No text could be extracted from this PDF.")

        # Load existing vectorstore from cache key (mutable workaround)
        existing_vs = st.session_state.get("vector_store")
        if existing_vs is None:
            vs = FAISS.from_documents(chunks, embeddings)
        else:
            existing_vs.add_documents(chunks)
            vs = existing_vs

        vs.save_local(VECTORSTORE_PATH)
        return vs, len(documents), len(chunks)

    finally:
        os.unlink(tmp_path)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; color:#7C3AED; margin-bottom:0.2rem; }
    .sub-header  { color:#6B7280; font-size:0.95rem; margin-bottom:1.5rem; }
    .source-box  {
        background:#2D2D3F; border-left:3px solid #7C3AED;
        padding:0.6rem 1rem; border-radius:4px;
        margin-bottom:0.5rem; font-size:0.85rem;
        color:#E5E7EB !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "qa_chain"     not in st.session_state: st.session_state.qa_chain     = None
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = []

# ── Load models ────────────────────────────────────────────────────────────────
embeddings, llm = load_models()

# Load vectorstore from disk if exists and session is fresh
if st.session_state.vector_store is None:
    vs = load_vectorstore(embeddings)
    if vs:
        st.session_state.vector_store = vs
        st.session_state.qa_chain = build_qa_chain(llm, vs)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Document Q&A")
    st.markdown("---")

    # Status
    has_docs = len(st.session_state.uploaded_docs) > 0
    vs_ready = st.session_state.vector_store is not None
    st.markdown(f"**Models:** ✅ Loaded")
    st.markdown(
        f"**Vector Store:** {'✅ Ready' if vs_ready else '⚠️ Empty'}",
    )
    st.markdown(
        f"**QA Chain:** {'✅ Ready' if st.session_state.qa_chain else '⚠️ Not ready'}",
    )

    st.markdown("---")
    st.markdown("### 📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF (max 200 MB)", type=["pdf"], label_visibility="collapsed"
    )
    top_k = st.slider("Sources to retrieve (top-k)", 1, 10, 3)

    if uploaded_file:
        st.info(f"**{uploaded_file.name}** — {uploaded_file.size/1024:.1f} KB")
        if st.button("🚀 PROCESS THIS PDF NOW", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF… this may take a minute…"):
                try:
                    vs, pages, chunks = process_pdf(
                        uploaded_file.getvalue(), uploaded_file.name, embeddings
                    )
                    st.session_state.vector_store = vs
                    st.session_state.qa_chain = build_qa_chain(llm, vs)
                    st.session_state.uploaded_docs.append({
                        "filename": uploaded_file.name,
                        "pages": pages,
                        "chunks": chunks,
                    })
                    st.session_state.messages = []
                    st.success(f"✅ Indexed **{pages} pages** → {chunks} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # Doc list
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown(f"### 📚 Indexed ({len(st.session_state.uploaded_docs)})")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"- **{doc['filename']}** — {doc['pages']}p, {doc['chunks']} chunks")

        if st.button("🗑 Reset / Clear All", use_container_width=True):
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.uploaded_docs = []
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("FastAPI · LangChain · Groq LLaMA3 · FAISS\nBy [Mohammad Ayesha Summaiyya](https://github.com/Ayesha037)")

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>📄 RAG Document Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload any PDF → ask questions → get source-attributed answers</div>", unsafe_allow_html=True)

has_docs = st.session_state.vector_store is not None

if not has_docs:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1**\nClick *Browse files* in the sidebar and select a PDF.")
    c2.info("**Step 2**\nClick **🚀 PROCESS THIS PDF NOW** to index it.")
    c3.info("**Step 3**\nType any question in the chat below.")

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    for q in [
        "What are the main topics covered?",
        "Summarise this document in 3 bullet points.",
        "What does chapter 2 discuss?",
        "Who is mentioned most frequently?",
        "What are the key findings or conclusions?",
    ]:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

st.markdown("### 💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"<div class='source-box'><b>Source {i}</b> — Page {src.get('page','?')}<br>{src.get('content','')}</div>",
                        unsafe_allow_html=True,
                    )

placeholder = "Ask a question about your document…" if has_docs else "Upload and process a PDF first to enable chat…"
user_input = st.chat_input(placeholder, disabled=not has_docs)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.qa_chain({"query": user_input})
                answer = result.get("result", "No answer generated.")
                source_docs = result.get("source_documents", [])
                sources = [
                    {
                        "content": doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""),
                        "page": doc.metadata.get("page", "unknown"),
                    }
                    for doc in source_docs[:top_k]
                ]
                st.markdown(answer)
                if sources:
                    with st.expander(f"📎 {len(sources)} source(s)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"<div class='source-box'><b>Source {i}</b> — Page {src.get('page','?')}<br>{src.get('content','')}</div>",
                                unsafe_allow_html=True,
                            )
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })
            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err, "sources": []
                })