"""
vectorstore_utils.py
---------------------
Step 4 of the pipeline: Build, save, and load a FAISS vector index
using free local HuggingFace embeddings.

Usage (standalone, builds index from a PDF):
    python vectorstore_utils.py path/to/document.pdf
"""

import sys
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR
from ingest import load_pdf, split_documents


def get_embedding_model():
    """
    Returns a free, local sentence-transformers embedding model.
    No API key required — runs entirely on CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, save_path: str = VECTORSTORE_DIR):
    """
    Build a FAISS vector index from document chunks and persist it to disk.
    """
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"[vectorstore] Saved FAISS index with {len(chunks)} vectors to '{save_path}'")
    return vectorstore


def load_vectorstore(save_path: str = VECTORSTORE_DIR):
    """
    Load a previously saved FAISS vector index from disk.
    """
    if not os.path.exists(save_path) or not os.listdir(save_path):
        raise FileNotFoundError(
            f"No vector index found at '{save_path}'. "
            f"Run ingestion first (e.g. `python vectorstore_utils.py <pdf>`)."
        )

    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[vectorstore] Loaded FAISS index from '{save_path}'")
    return vectorstore


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vectorstore_utils.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
