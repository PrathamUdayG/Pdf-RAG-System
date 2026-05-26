import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# ── Load environment variables ──────────────────────────────────────────
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """Retrieve a secret from Streamlit Cloud secrets or local .env file."""
    # Try Streamlit Cloud secrets first
    try:
        val = st.secrets.get(key, None)
        if val:
            return str(val)
    except Exception:
        pass
    # Fallback to environment variable (.env file)
    return os.getenv(key, default)

# Ensure API keys are available as env vars (needed by langchain / tavily)
for _key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    _val = get_secret(_key)
    if _val:
        os.environ[_key] = _val

# ── Page configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A — RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & global ── */
html, body, .stApp, .stMarkdown, .stButton, .stTextInput,
.stSlider, .stSelectbox, .stExpander, .stChatMessage {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 960px;
}

/* ── Fix: prevent Material Icon text from showing as plain text ── */
[data-testid="stFileUploader"] button[kind="secondary"] span[data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"],
[data-testid="baseButton-secondary"] span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    -webkit-text-fill-color: initial !important;
    font-size: 1.2rem !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero h1 {
    font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 0.95rem; font-weight: 300; margin: 0; }

/* ── Badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.82rem; font-weight: 500;
    padding: 0.35rem 0.9rem; border-radius: 20px;
    margin-bottom: 0.8rem;
}
.badge-ok   { background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.3);  color: #34d399; }
.badge-info { background: rgba(96,165,250,0.1);  border: 1px solid rgba(96,165,250,0.3);  color: #60a5fa; }

/* ── Sidebar stats ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; text-align: center;
}
.stat-card .val {
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-card .lbl {
    font-size: 0.7rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* ── Source chips ── */
.src-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 0.65rem 0.9rem;
    margin-bottom: 0.45rem; font-size: 0.82rem;
    color: #cbd5e1; line-height: 1.55;
}
.src-chip .tag {
    font-weight: 600; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.src-chip .tag-pdf { color: #34d399; }
.src-chip .tag-web { color: #f59e0b; }

/* ── Feature cards (empty state) ── */
.feat-card {
    text-align: center; padding: 1.4rem 1rem;
    border-radius: 12px;
}
.feat-card .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.feat-card .title { font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.feat-card .desc  { font-size: 0.78rem; color: #94a3b8; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167,139,250,0.25);
    border-radius: 12px; padding: 0.8rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(167,139,250,0.55); }

/* ── Sidebar background ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e1b4b);
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Hero ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>📄 PDF Question Answering</h1>
    <p>Upload a document · Ask anything · Get grounded answers powered by RAG + Web Search</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ──────────────────────────────────────────────
_defaults = {
    "vectorstore": None,
    "chat_history": [],          # list[dict] – {role, content, sources}
    "doc_stats": {},
    "processed_file_name": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes, file_name: str):
    """Parse PDF → chunk → embed → FAISS."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
        texts = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        vs = FAISS.from_documents(texts, embeddings)
        return vs, {"pages": len(docs), "chunks": len(texts), "file_name": file_name}
    finally:
        os.unlink(tmp_path)


def tavily_search(query: str, max_results: int = 3):
    """Run a Tavily web search. Returns list[dict] with title, url, content."""
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max_results, search_depth="basic")
        return resp.get("results", [])
    except Exception:
        return []


def _escape_html(text: str) -> str:
    """Minimal HTML escape so user/LLM text doesn't break the page."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_chat_history_text() -> str:
    """Compile the last few turns into a text block for the LLM."""
    turns = st.session_state.chat_history[-6:]  # last 3 pairs max
    parts = []
    for msg in turns:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts)


def get_answer(vectorstore, question: str, k: int = 4, use_web: bool = False):
    """
    Retrieve relevant chunks, optionally run a web search,
    then ask Gemini for a grounded answer with conversation memory.
    """
    # 1. PDF retrieval
    pdf_docs = vectorstore.similarity_search(question, k=k)
    pdf_context = "\n\n".join(
        f"[PDF Page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in pdf_docs
    )

    # 2. Web search (if enabled & key present)
    web_results = []
    web_context = ""
    if use_web:
        web_results = tavily_search(question)
        if web_results:
            web_context = "\n\n".join(
                f"[Web: {r.get('title', '')}]\n{r.get('content', '')}" for r in web_results
            )

    # 3. Conversation history
    history_text = build_chat_history_text()

    # 4. Build prompt
    prompt = f"""You are an expert document analyst and research assistant.
You are having a multi-turn conversation with the user. Use the conversation 
history below to understand context and follow-ups.

CONVERSATION HISTORY:
{history_text if history_text else "(This is the start of the conversation.)"}

DOCUMENT CONTEXT (from the uploaded PDF):
{pdf_context}
"""
    if web_context:
        prompt += f"""
WEB SEARCH RESULTS (supplementary information from the internet):
{web_context}
"""
    prompt += f"""
USER'S CURRENT QUESTION:
{question}

INSTRUCTIONS:
- Answer the question primarily using the DOCUMENT CONTEXT.
- If web search results are available, use them to enrich or validate your answer.
  When citing web information, mention it comes from external sources.
- If the document context is insufficient and no web results help, say so honestly.
- Keep track of the conversation history for follow-up questions.
- Use bullet points, numbered lists, or headings when they aid clarity.
- Be concise but thorough.

ANSWER:
"""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=1024)
    response = llm.invoke(prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)

    return answer_text, pdf_docs, web_results


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    k_value = st.slider(
        "Chunks to retrieve (k)", 1, 10, 4,
        help="Number of most-similar PDF chunks sent to the LLM.",
    )

    tavily_available = bool(get_secret("TAVILY_API_KEY"))
    web_search_on = st.toggle(
        "🌐 Web Search (Tavily)",
        value=False,
        disabled=not tavily_available,
        help="Supplement PDF answers with live web search results."
             + ("" if tavily_available else " ⚠️ Add TAVILY_API_KEY to your .env file to enable."),
    )

    if not tavily_available:
        st.caption("⚠️ Add `TAVILY_API_KEY` to `.env` to enable web search.")

    st.markdown("---")

    # Document stats
    if st.session_state.doc_stats:
        s = st.session_state.doc_stats
        st.markdown("### 📊 Document")
        st.markdown(
            f'<div class="stat-card"><div class="val">{s["pages"]}</div>'
            f'<div class="lbl">Pages</div></div>'
            f'<div class="stat-card"><div class="val">{s["chunks"]}</div>'
            f'<div class="lbl">Chunks</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f'📁 {s["file_name"]}')

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        '<div style="text-align:center;color:#64748b;font-size:0.72rem;padding-top:1rem;">'
        "LangChain · FAISS · Gemini · Tavily<br>Sentence-Transformers · Streamlit</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📎 Upload a PDF document", type="pdf",
    help="Drag-and-drop or click to browse.",
)

if uploaded_file is not None:
    # ── Process new file ────────────────────────────────────────────────
    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("🔍 Reading PDF & building vector index…"):
            file_bytes = uploaded_file.read()
            vs, stats = build_vectorstore(file_bytes, uploaded_file.name)
            st.session_state.vectorstore = vs
            st.session_state.doc_stats = stats
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.chat_history = []
        st.rerun()

    # ── Document-ready badge ────────────────────────────────────────────
    badges = (
        f'<span class="badge badge-ok">✅ {st.session_state.doc_stats["file_name"]} '
        f'— {st.session_state.doc_stats["pages"]} pages, '
        f'{st.session_state.doc_stats["chunks"]} chunks</span>'
    )
    if web_search_on:
        badges += '  <span class="badge badge-info">🌐 Web Search ON</span>'
    st.markdown(badges, unsafe_allow_html=True)

    # ── Render chat history using native Streamlit chat ─────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg["role"] == "assistant" and (msg.get("pdf_sources") or msg.get("web_sources")):
                with st.expander("📚 View sources"):
                    # PDF sources
                    if msg.get("pdf_sources"):
                        for i, src in enumerate(msg["pdf_sources"], 1):
                            pg = src.metadata.get("page", 0)
                            snippet = _escape_html(src.page_content[:400])
                            st.markdown(
                                f'<div class="src-chip">'
                                f'<div class="tag tag-pdf">📄 PDF · Chunk {i} · Page {int(pg) + 1}</div>'
                                f'{snippet}{"…" if len(src.page_content) > 400 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    # Web sources
                    if msg.get("web_sources"):
                        for r in msg["web_sources"]:
                            title = _escape_html(r.get("title", "Web result"))
                            url = r.get("url", "")
                            snippet = _escape_html(r.get("content", "")[:300])
                            st.markdown(
                                f'<div class="src-chip">'
                                f'<div class="tag tag-web">🌐 Web · {title}</div>'
                                f'{snippet}…<br>'
                                f'<a href="{url}" target="_blank" '
                                f'style="color:#60a5fa;font-size:0.75rem;">{url}</a>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ── Chat input ──────────────────────────────────────────────────────
    user_input = st.chat_input("💬 Ask a question about your PDF…")

    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Show user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    answer, pdf_sources, web_sources = get_answer(
                        st.session_state.vectorstore,
                        user_input,
                        k=k_value,
                        use_web=web_search_on,
                    )
                    st.markdown(answer)

                    # Append assistant message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "pdf_sources": pdf_sources,
                        "web_sources": web_sources,
                    })

                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err,
                    })

else:
    # ── Empty state — feature cards ─────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📤", "Upload", "Drop any PDF", "rgba(167,139,250,0.07)", "rgba(167,139,250,0.15)"),
        ("🔎", "Search", "Semantic retrieval", "rgba(96,165,250,0.07)", "rgba(96,165,250,0.15)"),
        ("🌐", "Web", "Tavily web search", "rgba(245,158,11,0.07)", "rgba(245,158,11,0.15)"),
        ("✨", "Answer", "Gemini-powered", "rgba(52,211,153,0.07)", "rgba(52,211,153,0.15)"),
    ]
    for col, (icon, title, desc, bg, bdr) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="feat-card" style="background:{bg};border:1px solid {bdr};">'
                f'<div class="icon">{icon}</div>'
                f'<div class="title">{title}</div>'
                f'<div class="desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )