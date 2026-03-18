"""Streamlit front-end for the ACME Customer Support RAG Assistant."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACME Support Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 ACME Customer Support Assistant")
st.markdown(
    "Ask any question about ACME's products, billing, or technical setup. "
    "Answers are grounded in the official knowledge base with enterprise-ready citations."
)

DEMO_PROMPTS = [
    "How do I reset my password?",
    "What is included in the Enterprise plan?",
    "What SLA uptime does ACME provide?",
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your OpenAI API key. Stored in session only.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )

    k = st.slider("Documents to retrieve (k)", min_value=1, max_value=8, value=4)

    st.divider()
    st.subheader("🎬 Demo mode")
    demo_prompt = st.selectbox("Pick a sample enterprise question", DEMO_PROMPTS, index=0)
    run_demo = st.button("▶ Run demo question")

    st.divider()

    rebuild = st.button("🔄 Rebuild Index", help="Re-embed all documents in data/docs/")

    st.divider()
    st.markdown(
        "**Knowledge base documents:**\n"
        + "\n".join(
            f"- `{p.name}`"
            for p in (Path(__file__).parent / "data" / "docs").glob("*.txt")
        )
    )

# ── Session state ──────────────────────────────────────────────────────────────
INDEX_PATH = str(Path(__file__).parent / "faiss_index")

if "pipeline" not in st.session_state or rebuild:
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to continue.")
        st.stop()

    with st.spinner("Loading knowledge base…"):
        pipeline = RAGPipeline(
            index_path=INDEX_PATH,
            model_name=model_name,
            k=k,
        )
        if rebuild or not Path(INDEX_PATH).exists():
            pipeline.build_index()
        else:
            pipeline.load_index()

    st.session_state["pipeline"] = pipeline
    st.session_state["model_name"] = model_name
    st.session_state["k"] = k

elif st.session_state.get("model_name") != model_name or st.session_state.get("k") != k:
    # Reload chain if settings changed
    pipeline: RAGPipeline = st.session_state["pipeline"]
    pipeline.model_name = model_name
    pipeline.k = k
    pipeline.load_index()
    st.session_state["model_name"] = model_name
    st.session_state["k"] = k

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about ACME…")
if run_demo:
    prompt = demo_prompt

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state["pipeline"].query(prompt)
                answer = result["answer"]
                sources = result["sources"]
                citations = result.get("citations", [])
            except Exception as exc:
                answer = f"❌ Error: {exc}"
                sources = []
                citations = []

        st.markdown(answer)

        if citations:
            with st.expander("📚 Enterprise citations"):
                for citation in citations:
                    source_name = Path(citation.get("source", "unknown")).name
                    page = citation.get("page")
                    page_info = f" · page {page}" if page is not None else ""
                    st.markdown(f"**[{citation['id']}]** `{source_name}`{page_info}")
                    if citation.get("excerpt"):
                        st.caption(citation["excerpt"])
        elif sources:
            with st.expander("📄 Source documents"):
                seen = set()
                for meta in sources:
                    src = meta.get("source", "unknown")
                    if src not in seen:
                        seen.add(src)
                        st.markdown(f"- `{Path(src).name}`")

    st.session_state["messages"].append({"role": "assistant", "content": answer})
