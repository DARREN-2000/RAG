# 🤖 ACME Customer Support RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** application that turns plain-text
knowledge-base documents into an intelligent customer support chatbot.

> **Use-case:** Customer support teams upload product FAQs, technical guides, and getting-started
> documentation. Users then ask natural-language questions and receive accurate, source-cited
> answers drawn entirely from those documents — no hallucination, no out-of-scope answers.

---

## Architecture

```
User question
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│                       RAGPipeline                        │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │ DocumentLoader│──▶│ Text Splitter│──▶│  Embeddings│  │
│  │  (.txt/.pdf) │   │  (chunks)    │   │  (OpenAI)  │  │
│  └──────────────┘   └──────────────┘   └─────┬──────┘  │
│                                               │          │
│                                               ▼          │
│                                        ┌──────────────┐  │
│                                        │  FAISS Index │  │
│                                        └──────┬───────┘  │
│                                               │ retrieve  │
│                                               ▼          │
│                                        ┌──────────────┐  │
│                                        │  ChatOpenAI  │  │
│                                        │  (GPT-4o-mini│  │
│                                        └──────┬───────┘  │
└──────────────────────────────────────────────┼──────────┘
                                               │
                                               ▼
                                    Answer + source citations
```

**Tech stack**

| Component | Library |
|-----------|---------|
| LLM | OpenAI GPT-4o-mini (configurable) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | FAISS (local, no server needed) |
| RAG framework | LangChain |
| Web UI | Streamlit |
| Document formats | Plain text (`.txt`) · PDF (`.pdf`) |

---

## Quick Start

### 1 — Clone and install

```bash
git clone https://github.com/DARREN-2000/RAG.git
cd RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Add your OpenAI API key

```bash
cp .env.example .env        # then edit .env and set OPENAI_API_KEY=sk-...
```

Or export it directly:

```bash
export OPENAI_API_KEY="sk-..."
```

### 3 — (Optional) Add your own documents

Drop `.txt` or `.pdf` files into `data/docs/`. Three sample documents are included:

| File | Contents |
|------|----------|
| `product_faq.txt` | Product features, pricing, and plans |
| `technical_support.txt` | Password reset, billing, security, SLAs |
| `getting_started.txt` | Step-by-step onboarding guide |

### 4 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

![screenshot placeholder](https://placehold.co/800x450?text=ACME+Support+Assistant+UI)

### Built-in demo flow

Use **🎬 Demo mode** in the sidebar to run sample enterprise support questions in one click.
Each answer includes a **📚 Enterprise citations** section with numbered sources, optional page
metadata, and supporting excerpts.

### 5 — Use the Python API directly

```python
from rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.build_index()   # embed docs and persist FAISS index (run once)

result = pipeline.query("How do I reset my password?")
print(result["answer"])
# → "On the login page click 'Forgot Password'…"

for src in result["sources"]:
    print(src["source"])
```

---

## Project Structure

```
RAG/
├── app.py                  # Streamlit web UI
├── requirements.txt
├── .gitignore
├── data/
│   └── docs/               # Knowledge-base documents (add yours here)
│       ├── product_faq.txt
│       ├── technical_support.txt
│       └── getting_started.txt
├── rag/
│   ├── __init__.py
│   ├── pipeline.py         # High-level RAGPipeline class
│   ├── document_loader.py  # Load & split .txt / .pdf files
│   ├── vector_store.py     # FAISS build / load helpers
│   └── chain.py            # LangChain RetrievalQA chain
└── tests/
    └── test_rag.py         # Unit tests (no API calls needed)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests run with mocked dependencies — **no OpenAI API key is needed** to run the test suite.

---

## Configuration

All parameters have sensible defaults but can be overridden via `RAGPipeline` constructor
arguments or via the Streamlit sidebar at runtime:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `docs_dir` | `data/docs/` | Directory containing knowledge-base files |
| `index_path` | `faiss_index/` | Where to persist the FAISS index |
| `model_name` | `gpt-4o-mini` | OpenAI chat model |
| `chunk_size` | `500` | Characters per document chunk |
| `chunk_overlap` | `50` | Overlap between consecutive chunks |
| `k` | `4` | Number of documents retrieved per query |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## License

MIT
