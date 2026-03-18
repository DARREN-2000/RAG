"""High-level RAG pipeline that wires together loading, indexing, and querying."""

from pathlib import Path
from typing import Any, Dict, List

from langchain.schema import Document

from rag.chain import build_qa_chain
from rag.document_loader import load_documents, split_documents
from rag.vector_store import build_vector_store, load_vector_store

DEFAULT_DOCS_DIR = str(Path(__file__).parent.parent / "data" / "docs")
DEFAULT_INDEX_PATH = str(Path(__file__).parent.parent / "faiss_index")
MAX_EXCERPT_CHARS = 220


class RAGPipeline:
    """End-to-end RAG pipeline for document-grounded question answering.

    Usage::

        pipeline = RAGPipeline()
        pipeline.build_index()            # only needed once
        result = pipeline.query("How do I reset my password?")
        print(result["answer"])
    """

    def __init__(
        self,
        docs_dir: str = DEFAULT_DOCS_DIR,
        index_path: str = DEFAULT_INDEX_PATH,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 4,
    ) -> None:
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self._chain = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Load documents, create embeddings, and persist the FAISS index."""
        docs = load_documents(self.docs_dir)
        chunks = split_documents(docs, self.chunk_size, self.chunk_overlap)
        vector_store = build_vector_store(chunks, persist_path=self.index_path)
        self._chain = build_qa_chain(vector_store, model_name=self.model_name, k=self.k)

    def load_index(self) -> None:
        """Load an existing FAISS index from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"Index not found at '{self.index_path}'. "
                "Run build_index() first."
            )
        vector_store = load_vector_store(self.index_path)
        self._chain = build_qa_chain(vector_store, model_name=self.model_name, k=self.k)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict:
        """Answer a question using the RAG pipeline.

        Args:
            question: Natural-language question to answer.

        Returns:
            A dict with keys ``"answer"`` (str) and ``"sources"`` (list of
            Document metadata dicts).

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        if self._chain is None:
            raise RuntimeError(
                "No index loaded. Call build_index() or load_index() first."
            )
        result = self._chain.invoke({"query": question})
        source_documents = result.get("source_documents", [])
        sources = [doc.metadata for doc in source_documents]
        citations = self._build_citations(source_documents)
        return {"answer": result["result"], "sources": sources, "citations": citations}

    @staticmethod
    def _build_citations(source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Build stable, UI-friendly citation entries from source documents."""
        citations: List[Dict[str, Any]] = []
        for idx, doc in enumerate(source_documents, start=1):
            metadata = doc.metadata or {}
            excerpt = " ".join(doc.page_content.split()) if doc.page_content else ""
            citations.append(
                {
                    "id": idx,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page"),
                    "excerpt": excerpt[:MAX_EXCERPT_CHARS],
                }
            )
        return citations
