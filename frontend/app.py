import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f, #2563eb);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .citation-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
        color: #1e40af;
        margin-top: 8px;
    }
    .status-ready {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85em;
    }
    .status-not-ready {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85em;
    }
    .upload-instruction {
        background-color: #fef9c3;
        border-left: 4px solid #eab308;
        padding: 10px;
        border-radius: 4px;
        color: #854d0e;
        margin: 8px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def upload_pdf(file):
    try:
        files = {
            "file": (file.name, file.getvalue(), "application/pdf")
        }
        r = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=300
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        return {"success": False, "error": r.json().get("detail")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_api(question: str, top_k: int = 5):
    try:
        r = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "top_k": top_k,
                "use_history": True
            },
            timeout=300
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        return {"success": False, "error": r.json().get("detail")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def reset_system():
    try:
        r = requests.get(f"{API_BASE_URL}/reset", timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def get_uploaded_files():
    try:
        r = requests.get(f"{API_BASE_URL}/files", timeout=5)
        return r.json().get("files", []) if r.status_code == 200 else []
    except Exception:
        return []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False

with st.sidebar:
    st.markdown("## 📚 RAG Document Q&A")
    st.markdown("---")

    st.markdown("### 🔌 System Status")
    health = check_api_health()

    if health:
        st.markdown(
            '<span class="status-ready">✅ API Connected</span>',
            unsafe_allow_html=True
        )
        if health.get("vectorstore_ready"):
            st.markdown(
                '<span class="status-ready">✅ Documents Ready</span>',
                unsafe_allow_html=True
            )
            st.session_state.documents_uploaded = True
        else:
            st.markdown(
                '<span class="status-not-ready">⚠️ No Documents Loaded</span>',
                unsafe_allow_html=True
            )
            st.session_state.documents_uploaded = False
    else:
        st.markdown(
            '<span class="status-not-ready">❌ API Offline</span>',
            unsafe_allow_html=True
        )
        st.error(
            "Backend not running!\n\n"
            "Open CMD and run:\n"
            "`python -m app.main`"
        )

    st.markdown("---")

    st.markdown("### 📄 Upload Any PDF")
    st.markdown(
        "**Step 1:** Select your PDF file below"
    )

    uploaded_file = st.file_uploader(
        "Choose any PDF file",
        type=["pdf"],
        help="Supports any PDF — textbooks, resumes, reports, stories, etc."
    )

    if uploaded_file is not None:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        file_size_mb = file_size_kb / 1024

        if file_size_mb >= 1:
            size_str = f"{file_size_mb:.1f} MB"
        else:
            size_str = f"{file_size_kb:.1f} KB"

        st.success(f"✅ **{uploaded_file.name}** selected!")
        st.info(f"📊 Size: {size_str}")

        st.markdown(
            '<div class="upload-instruction">'
            'Step 2: Click the button below to process!'
            '</div>',
            unsafe_allow_html=True
        )

        process_clicked = st.button(
            "🚀 PROCESS THIS PDF NOW",
            use_container_width=True,
            type="primary"
        )

        if process_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📖 Reading PDF...")
            progress_bar.progress(20)

            with st.spinner(
                f"Processing '{uploaded_file.name}'...\n"
                "This may take 1-2 minutes for large files."
            ):
                result = upload_pdf(uploaded_file)

            progress_bar.progress(100)

            if result["success"]:
                data = result["data"]
                status_text.empty()
                progress_bar.empty()

                st.success(
                    f"🎉 **Successfully Processed!**\n\n"
                    f"📄 File: {data['filename']}\n\n"
                    f"📑 Pages: {data['pages']}\n\n"
                    f"🧩 Chunks: {data['chunks']}\n\n"
                    f"💾 Size: {data['file_size']}"
                )
                st.balloons()
                st.session_state.documents_uploaded = True
                st.session_state.messages = []
                st.rerun()
            else:
                status_text.empty()
                progress_bar.empty()
                st.error(
                    f"❌ Processing failed!\n\n"
                    f"Error: {result['error']}\n\n"
                    "Try again or use a different PDF."
                )

    st.markdown("---")

    files = get_uploaded_files()
    if files:
        st.markdown("### 📂 Loaded Documents")
        for f in files:
            st.markdown(f"✅ 📄 {f}")
        st.caption(
            f"{len(files)} document(s) ready for questions"
        )

    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks = more context but slower response"
    )

    st.markdown("---")

    st.markdown("### 🔄 Reset")
    st.caption("Clear all documents and start fresh")
    if st.button(
        "🗑️ Reset Everything",
        use_container_width=True
    ):
        with st.spinner("Resetting system..."):
            if reset_system():
                st.session_state.messages = []
                st.session_state.documents_uploaded = False
                st.success("✅ Reset complete!")
                st.rerun()
            else:
                st.error("❌ Reset failed!")

    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    **RAG Chatbot** v1.0

    Works with ANY PDF:
    - 📚 Textbooks
    - 📰 Research papers
    - 📋 Reports
    - 📖 Stories
    - 📄 Resumes
    - And more!

    Built with:
    - 🦜 LangChain
    - 🔍 FAISS
    - ⚡ Groq LLaMA3
    - 🚀 FastAPI
    - 🎈 Streamlit
    """)
 
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Document Q&A Chatbot</h1>
    <p>Upload ANY PDF and ask questions in natural language.
    Get accurate answers with source citations — powered by AI!</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.documents_uploaded:
    st.info(
        "👈 **Get Started:** Select a PDF in the sidebar "
        "and click **'🚀 PROCESS THIS PDF NOW'** to begin!"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📄 Step 1: Upload
        Click **Browse files** in
        the sidebar and select
        any PDF document.
        """)
    with col2:
        st.markdown("""
        ### ⚡ Step 2: Process
        Click the big
        **🚀 PROCESS THIS PDF NOW**
        button in the sidebar.
        """)
    with col3:
        st.markdown("""
        ### 💬 Step 3: Ask!
        Type any question about
        your document in the
        chat box below.
        """)

    st.markdown("---")

else:
    files = get_uploaded_files()
    if files:
        st.success(
            f"✅ **{len(files)} document(s) loaded and ready!** "
            f"Ask any question below."
        )

st.markdown("### 💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("citations"):
                with st.expander("📎 View Sources"):
                    for cite in message["citations"]:
                        st.markdown(
                            f"""<div class="citation-box">
                            📄 <b>{cite['source']}</b>
                            — Page {cite['page']}<br>
                            <i>{cite.get('preview', '')[:150]}</i>
                            </div>""",
                            unsafe_allow_html=True
                        )
            if message.get("model"):
                st.caption(f"🤖 Powered by {message['model']}")

placeholder_text = (
    "Ask a question about your documents..."
    if st.session_state.documents_uploaded
    else "Upload and process a PDF first to enable chat..."
)

if prompt := st.chat_input(
    placeholder_text,
    disabled=not st.session_state.documents_uploaded
):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = query_api(prompt, top_k=top_k)

        if result["success"]:
            data = result["data"]
            answer = data["answer"]

            st.markdown(answer)

            if data.get("citations"):
                with st.expander("📎 View Sources"):
                    for cite in data["citations"]:
                        st.markdown(
                            f"""<div class="citation-box">
                            📄 <b>{cite['source']}</b>
                            — Page {cite['page']}<br>
                            <i>{cite.get('preview', '')[:150]}</i>
                            </div>""",
                            unsafe_allow_html=True
                        )

            st.caption(f"🤖 Powered by {data.get('model', '')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": data.get("citations", []),
                "model": data.get("model", "")
            })

        else:
            error = f"❌ Error: {result['error']}"
            st.error(error)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error
            })