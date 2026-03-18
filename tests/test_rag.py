"""Unit tests for the RAG pipeline (no real API calls required)."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the rag package can be imported without installing heavy
# dependencies (langchain, openai, faiss, etc.) in the test environment.
# ---------------------------------------------------------------------------

def _make_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


# Build a fake Document class
class FakeDocument:
    def __init__(self, page_content: str = "", metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Stub out all third-party modules before importing rag.*
_stubs = {
    "langchain": _make_stub("langchain"),
    "langchain.schema": _make_stub("langchain.schema", {"Document": FakeDocument}),
    "langchain.text_splitter": _make_stub(
        "langchain.text_splitter",
        {"RecursiveCharacterTextSplitter": MagicMock(return_value=MagicMock())},
    ),
    "langchain.chains": _make_stub("langchain.chains", {"RetrievalQA": MagicMock()}),
    "langchain.prompts": _make_stub(
        "langchain.prompts", {"PromptTemplate": MagicMock(return_value=MagicMock())}
    ),
    "langchain_community": _make_stub("langchain_community"),
    "langchain_community.document_loaders": _make_stub(
        "langchain_community.document_loaders",
        {
            "DirectoryLoader": MagicMock(),
            "PyPDFLoader": MagicMock(),
            "TextLoader": MagicMock(),
        },
    ),
    "langchain_community.vectorstores": _make_stub(
        "langchain_community.vectorstores", {"FAISS": MagicMock()}
    ),
    "langchain_openai": _make_stub(
        "langchain_openai",
        {"OpenAIEmbeddings": MagicMock(), "ChatOpenAI": MagicMock()},
    ),
}

for mod_name, mod in _stubs.items():
    sys.modules.setdefault(mod_name, mod)

# Now it is safe to import our package
from rag.document_loader import load_documents, split_documents  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Tests for document_loader
# ---------------------------------------------------------------------------

SAMPLE_DOCS_DIR = str(Path(__file__).parent.parent / "data" / "docs")


def test_load_documents_returns_documents(tmp_path):
    """load_documents should return at least one Document per .txt file."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, world!", encoding="utf-8")

    fake_doc = FakeDocument("Hello, world!", {"source": str(txt_file)})

    with patch("rag.document_loader.TextLoader") as MockLoader:
        instance = MockLoader.return_value
        instance.load.return_value = [fake_doc]

        docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].page_content == "Hello, world!"


def test_load_documents_raises_for_missing_dir():
    """load_documents should raise FileNotFoundError for a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_documents("/non/existent/path")


def test_load_documents_raises_for_empty_dir(tmp_path):
    """load_documents should raise ValueError when no supported files are present."""
    with pytest.raises(ValueError, match="No supported documents"):
        load_documents(str(tmp_path))


def test_split_documents_produces_chunks():
    """split_documents should call the splitter and return chunked documents."""
    docs = [FakeDocument("a" * 1000, {})]
    fake_chunks = [FakeDocument("a" * 200, {}) for _ in range(5)]

    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = fake_chunks

    with patch("rag.document_loader.RecursiveCharacterTextSplitter", return_value=mock_splitter):
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

    assert chunks == fake_chunks
    mock_splitter.split_documents.assert_called_once_with(docs)


def test_split_documents_respects_chunk_size():
    """RecursiveCharacterTextSplitter should be created with the supplied parameters."""
    docs = [FakeDocument("x" * 100, {})]

    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = []

    with patch(
        "rag.document_loader.RecursiveCharacterTextSplitter", return_value=mock_splitter
    ) as MockSplitter:
        split_documents(docs, chunk_size=300, chunk_overlap=30)

    MockSplitter.assert_called_once_with(
        chunk_size=300, chunk_overlap=30, length_function=len
    )


# ---------------------------------------------------------------------------
# Tests for RAGPipeline
# ---------------------------------------------------------------------------

def test_pipeline_query_raises_without_index():
    """query() must raise RuntimeError before build_index/load_index is called."""
    pipeline = RAGPipeline()
    with pytest.raises(RuntimeError, match="No index loaded"):
        pipeline.query("What is ACME?")


def test_pipeline_load_index_raises_for_missing_path(tmp_path):
    """load_index() must raise FileNotFoundError when the index does not exist."""
    pipeline = RAGPipeline(index_path=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError):
        pipeline.load_index()


def test_pipeline_query_returns_answer_and_sources():
    """query() should return a dict with 'answer' and 'sources' keys."""
    pipeline = RAGPipeline()

    # Inject a mock chain
    mock_chain = MagicMock()
    source_doc = FakeDocument("some text", {"source": "faq.txt"})
    mock_chain.invoke.return_value = {
        "result": "You can reset your password from the login page.",
        "source_documents": [source_doc],
    }
    pipeline._chain = mock_chain

    result = pipeline.query("How do I reset my password?")

    assert result["answer"] == "You can reset your password from the login page."
    assert result["sources"] == [{"source": "faq.txt"}]
    mock_chain.invoke.assert_called_once_with({"query": "How do I reset my password?"})


def test_pipeline_build_index_wires_components(tmp_path):
    """build_index() should call load_documents, split_documents, and build_vector_store."""
    pipeline = RAGPipeline(docs_dir=str(tmp_path), index_path=str(tmp_path / "idx"))

    fake_doc = FakeDocument("content", {})
    fake_chunk = FakeDocument("chunk", {})
    fake_vs = MagicMock()
    fake_chain = MagicMock()

    with (
        patch("rag.pipeline.load_documents", return_value=[fake_doc]) as mock_load,
        patch("rag.pipeline.split_documents", return_value=[fake_chunk]) as mock_split,
        patch("rag.pipeline.build_vector_store", return_value=fake_vs) as mock_vs,
        patch("rag.pipeline.build_qa_chain", return_value=fake_chain) as mock_chain,
    ):
        pipeline.build_index()

    mock_load.assert_called_once_with(str(tmp_path))
    mock_split.assert_called_once_with([fake_doc], pipeline.chunk_size, pipeline.chunk_overlap)
    mock_vs.assert_called_once_with([fake_chunk], persist_path=str(tmp_path / "idx"))
    mock_chain.assert_called_once()
    assert pipeline._chain is fake_chain


def test_sample_docs_exist():
    """The bundled sample knowledge-base documents must be present."""
    docs_dir = Path(SAMPLE_DOCS_DIR)
    txt_files = list(docs_dir.glob("*.txt"))
    assert len(txt_files) >= 3, "Expected at least 3 sample .txt documents"


def test_sample_docs_non_empty():
    """Each sample document should contain a reasonable amount of text."""
    docs_dir = Path(SAMPLE_DOCS_DIR)
    for txt_file in docs_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        assert len(content) > 100, f"{txt_file.name} is too short to be useful"
