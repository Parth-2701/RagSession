"""
ingest.py
---------
Step 2 & 3 of the pipeline: Load a PDF and split it into chunks.

Usage (standalone):
    python ingest.py path/to/document.pdf
"""

import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(pdf_path: str):
    """
    Step 2: Load a PDF file and return a list of LangChain Document objects
    (one per page, with page metadata attached).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[ingest] Loaded {len(documents)} page(s) from '{pdf_path}'")
    return documents


def split_documents(documents):
    """
    Step 3: Split loaded documents into smaller overlapping chunks using
    RecursiveCharacterTextSplitter. This preserves semantic boundaries
    (paragraphs/sentences) as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    print("\n--- Sample chunk ---")
    print(chunks[0].page_content[:500])
    print("...")