import os
import queue
import subprocess
import sys
import threading

import streamlit as st
from pymilvus import MilvusClient

st.set_page_config(page_title="DocFusion", page_icon="🧠", layout="wide")

if os.name == "nt":
    VENV_PYTHON = os.path.join(sys.prefix, "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(sys.prefix, "bin", "python")

# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(path: str):
    with open(path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/main.css")

# ── Milvus ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_milvus_client():
    try:
        return MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    except Exception:
        return None

# ── Warning filter ────────────────────────────────────────────────────────────
IGNORED_WARNINGS = [
    "UserWarning", "pkg_resources is deprecated", "from pkg_resources import",
    "WARNING: All log messages before absl", "huggingface/tokenizers",
    "The current process just got forked", "Avoid using `tokenizers` before the fork",
    "Explicitly set the environment variable TOKENIZERS_PARALLELISM",
    "skipping fork() handlers", "I0000 00:00:", "[notice]", "warnings.warn",
    "UnsupportedFieldAttributeWarning", "validate_default", "_generate_schema.py",
]

def is_ignored_warning(line):
    return any(w in line for w in IGNORED_WARNINGS)

# ── Commands ──────────────────────────────────────────────────────────────────
def run_command(command, status_label="Working..."):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # force real-time stdout flushing

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, encoding="utf-8", errors="replace", env=env
    )

    out_q: queue.Queue = queue.Queue()
    err_q: queue.Queue = queue.Queue()

    def read_stream(stream, q):
        for line in stream:
            q.put(line)
        q.put(None)  # sentinel

    threading.Thread(target=read_stream, args=(process.stdout, out_q), daemon=True).start()
    threading.Thread(target=read_stream, args=(process.stderr, err_q), daemon=True).start()

    output_text = []
    error_text  = []
    out_done = err_done = False

    with st.status(status_label, expanded=True) as status:
        output_area = st.empty()
        error_area  = st.empty()

        while not (out_done and err_done):
            try:
                line = out_q.get(timeout=0.1)
                if line is None:
                    out_done = True
                else:
                    output_text.append(line)
                    output_area.text_area("Output", "".join(output_text), height=220)
            except queue.Empty:
                pass

            try:
                line = err_q.get_nowait()
                if line is None:
                    err_done = True
                elif not is_ignored_warning(line):
                    error_text.append(line)
                    error_area.text_area("Errors", "".join(error_text), height=120)
            except queue.Empty:
                pass

        process.wait()

        quota_hit = any("QUOTA_EXHAUSTED" in l or "RESOURCE_EXHAUSTED" in l or "quota" in l.lower() for l in error_text)

        if process.returncode == 2 or quota_hit:
            status.update(label="❌ API Quota Exhausted", state="error", expanded=False)
            st.error(
                "**Gemini API quota exhausted.**\n\n"
                "Your free daily limit has been reached. Options:\n"
                "- Wait until tomorrow (free tier resets daily)\n"
                "- Get a new API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
                "- Add billing to your Google Cloud project",
                icon="🚫",
            )
            return False
        elif process.returncode == 0 or not error_text:
            status.update(label="✅ Done!", state="complete", expanded=False)
            return True
        else:
            status.update(label="❌ Completed with errors", state="error", expanded=True)
            return False

def run_dump(pdfs, output_dir):
    if not pdfs or not output_dir:
        st.error("Please upload at least one PDF and specify an output directory.")
        return
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdf_paths = []
    for pdf in pdfs:
        pdf_path = os.path.join(output_dir, pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())
        pdf_paths.append(pdf_path)
    run_command([VENV_PYTHON, "automation.py", "dump", *pdf_paths, output_dir],
                status_label=f"⚙️ Processing {len(pdf_paths)} PDF(s)...")

def run_search(query):
    command = [VENV_PYTHON, "automation.py", "search"]
    if query:
        command.append(query)
    success = run_command(command, status_label=f"🔍 Searching & summarizing: \"{query}\"")

    if success:
        pdf_path = os.path.join("latex-output", "output.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.success("Your summary paper is ready!")
            st.download_button(
                label="📄 Download Summary PDF",
                data=pdf_bytes,
                file_name=f"docfusion_{query.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ── Navbar ────────────────────────────────────────────────────────────────────
client = get_milvus_client()
collections = client.list_collections() if client else []
db_badge = (
    '<span class="badge badge-green">● Milvus Connected</span>'
    if client else
    '<span class="badge badge-red">● Milvus Offline</span>'
)

st.markdown(f"""
<div class="navbar">
    <div>
        <div class="navbar-brand">🧠 DocFusion</div>
        <div class="navbar-tagline">AI-powered PDF Search &amp; Research Summarization</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
        {db_badge}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗄️ Collections")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if client:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">{len(collections)}</div>
                <div class="stat-label">Collections</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if collections:
            selected = st.selectbox("Select to delete", collections,
                                    index=None, placeholder="Choose a collection...")
            if st.button("🗑️ Delete Collection", use_container_width=True):
                st.session_state["delete_confirm"] = True

            if st.session_state.get("delete_confirm", False):
                st.warning(f"Delete **{selected}**?")
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete"):
                    client.drop_collection(collection_name=selected)
                    st.success("Deleted!")
                    del st.session_state["delete_confirm"]
                    st.rerun()
                if c2.button("Cancel"):
                    del st.session_state["delete_confirm"]
                    st.rerun()

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**Stored collections:**")
            for col in collections:
                st.markdown(f"- `{col}`")
        else:
            st.info("No collections yet. Upload and process PDFs to get started.")
    else:
        st.error("Milvus offline. Run:\n```\ndocker compose up -d\n```")

# ── Main area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📂</div>
        <div class="card-title">Upload & Process PDFs</div>
        <div class="card-desc">Upload one or more research papers. They will be parsed, embedded, and stored in the vector database for search.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_pdfs = st.file_uploader(
        "Drop PDFs here", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_pdfs:
        st.caption(f"✅ {len(uploaded_pdfs)} file(s) ready to process")

    output_directory = st.text_input("Output Directory", placeholder="e.g.  ouput-directory")
    if st.button("⚙️  Process PDFs", use_container_width=True):
        run_dump(uploaded_pdfs, output_directory)

with col_right:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🔍</div>
        <div class="card-title">Search & Summarize</div>
        <div class="card-desc">Enter a research topic or keyword. DocFusion will retrieve relevant sections and generate a structured summary paper.</div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("Topic", placeholder="e.g.  Convolutional Neural Networks", label_visibility="collapsed")

    if st.button("✨  Generate Summary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query first.")
        else:
            run_search(query)
