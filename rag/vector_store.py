"""Vector store creation and management using FAISS."""

import os
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vector_store(
    documents: List[Document],
    persist_path: Optional[str] = None,
) -> FAISS:
    """Create a FAISS vector store from documents.

    Args:
        documents: List of chunked Document objects.
        persist_path: Optional path to save the index to disk.

    Returns:
        A FAISS vector store instance.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    if persist_path:
        vector_store.save_local(persist_path)

    return vector_store


def load_vector_store(persist_path: str) -> FAISS:
    """Load an existing FAISS vector store from disk.

    .. warning::
        ``allow_dangerous_deserialization=True`` is required by LangChain because the FAISS
        index is persisted using Python's ``pickle`` format.  Only load indexes from
        directories you created yourself or that you fully trust.  Never load an index file
        received from an untrusted source, as a malicious pickle payload can execute
        arbitrary code on your machine.

    Args:
        persist_path: Path where the FAISS index was previously saved by
            :func:`build_vector_store`.  Must point to a trusted, locally created index.

    Returns:
        A FAISS vector store instance.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
