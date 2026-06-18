import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:7860")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #7C3AED;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6B7280;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .status-ok   { color: #10B981; font-weight: 600; }
    .status-warn { color: #F59E0B; font-weight: 600; }
    .status-err  { color: #EF4444; font-weight: 600; }
    .source-box {
        background: #1E1E2E;
        border-left: 3px solid #7C3AED;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False


# ── Helper: call backend ───────────────────────────────────────────────────────
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def upload_pdf(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the server running on port 7860?"}, 503
    except Exception as e:
        return {"error": str(e)}, 500


def ask_question(question: str, top_k: int = 3):
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend."}, 503
    except Exception as e:
        return {"error": str(e)}, 500


def list_documents():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        return r.json() if r.status_code == 200 else {"count": 0, "documents": []}
    except Exception:
        return {"count": 0, "documents": []}


def reset_system():
    try:
        r = requests.get(f"{API_URL}/reset", timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Document Q&A")
    st.markdown("---")

    # Health check
    health = check_health()
    if health:
        vs = health.get("vector_store", "empty")
        qa = health.get("qa_chain", "not_ready")
        st.markdown(f"**Backend:** <span class='status-ok'>● Online</span>", unsafe_allow_html=True)
        st.markdown(
            f"**Vector Store:** "
            f"<span class='{'status-ok' if vs == 'ready' else 'status-warn'}'>{'✅ Ready' if vs == 'ready' else '⚠ Empty'}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**QA Chain:** "
            f"<span class='{'status-ok' if qa == 'ready' else 'status-warn'}'>{'✅ Ready' if qa == 'ready' else '⚠ Not ready'}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("**Backend:** <span class='status-err'>● Offline</span>", unsafe_allow_html=True)
        st.warning("Start the backend: `python main.py`")

    st.markdown("---")
    st.markdown("### 📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF (max 200 MB)",
        type=["pdf"],
        label_visibility="collapsed",
    )

    top_k = st.slider("Sources to retrieve (top-k)", 1, 10, 3)

    if uploaded_file:
        st.info(f"**{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")
        if st.button("🚀 PROCESS THIS PDF NOW", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF… this may take a minute…"):
                result, status = upload_pdf(uploaded_file.getvalue(), uploaded_file.name)

            if status == 200:
                st.success(
                    f"✅ Indexed **{result.get('pages', '?')} pages** "
                    f"→ {result.get('chunks', '?')} chunks"
                )
                st.session_state.docs_ready = True
                st.session_state.messages = []  # fresh chat for new doc
            else:
                err = result.get("error", "Unknown error")
                st.error(f"❌ Upload failed\n\n`{err}`")

    # Uploaded docs list
    docs_info = list_documents()
    if docs_info["count"] > 0:
        st.markdown("---")
        st.markdown(f"### 📚 Indexed Documents ({docs_info['count']})")
        for doc in docs_info["documents"]:
            st.markdown(
                f"- **{doc['filename']}** — {doc['pages']} pages, {doc['chunks']} chunks"
            )
        if st.button("🗑 Reset / Clear All", use_container_width=True):
            res, status = reset_system()
            if status == 200:
                st.session_state.docs_ready = False
                st.session_state.messages = []
                st.success("System reset.")
                st.rerun()
            else:
                st.error("Reset failed.")

    st.markdown("---")
    st.caption("RAG Document Intelligence System\nFastAPI + LangChain + Groq LLaMA3 + FAISS")
    st.caption("By [Mohammad Ayesha Summaiyya](https://github.com/Ayesha037)")


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>📄 RAG Document Intelligence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Upload any PDF → ask questions in plain English → get source-attributed answers</div>",
    unsafe_allow_html=True,
)

# Steps guide (shown before any doc is uploaded)
docs_info = list_documents()
has_docs = docs_info["count"] > 0

if not has_docs:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nClick *Browse files* in the sidebar and select a PDF.")
    with col2:
        st.info("**Step 2**\nClick **🚀 PROCESS THIS PDF NOW** to index it.")
    with col3:
        st.info("**Step 3**\nType any question in the chat below.")

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    examples = [
        "What are the main topics covered?",
        "Summarise this document in 3 bullet points.",
        "What does chapter 2 discuss?",
        "Who is mentioned most frequently?",
        "What are the key findings or conclusions?",
    ]
    for q in examples:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Chat ───────────────────────────────────────────────────────────────────────
st.markdown("### 💬 Chat")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"<div class='source-box'><b>Source {i}</b> — Page {src.get('page', '?')}<br>{src.get('content', '')}</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
placeholder = (
    "Ask a question about your document…" if has_docs
    else "Upload and process a PDF first to enable chat…"
)
user_input = st.chat_input(placeholder, disabled=not has_docs)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result, status = ask_question(user_input, top_k=top_k)

        if status == 200:
            answer = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
            confidence = result.get("confidence", "low")

            st.markdown(answer)

            if sources:
                with st.expander(f"📎 {len(sources)} source(s) — confidence: {confidence}"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"<div class='source-box'><b>Source {i}</b> — Page {src.get('page', '?')}<br>{src.get('content', '')}</div>",
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            err = result.get("error", "Unknown error from backend.")
            st.error(f"❌ {err}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {err}",
                "sources": [],
            })