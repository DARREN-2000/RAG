"""Document loading and splitting utilities."""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader


def load_documents(source_dir: str) -> List[Document]:
    """Load all supported documents from a directory.

    Supports .txt and .pdf files.

    Args:
        source_dir: Path to the directory containing documents.

    Returns:
        A list of LangChain Document objects.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Document directory not found: {source_dir}")

    documents: List[Document] = []

    for txt_file in source_path.rglob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())

    for pdf_file in source_path.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No supported documents (.txt or .pdf) found in: {source_dir}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
