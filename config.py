"""
Central configuration for the RAG Chatbot.
Edit these values to tune chunking, retrieval, and model behavior.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

# --- Embedding model (free, runs locally via sentence-transformers) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Text splitting ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 4  # number of chunks to retrieve per query

# --- LLM (Google Gemini free tier) ---
GEMINI_MODEL_NAME = "gemini-1.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Prompt template (forces grounded answers, no hallucination) ---
QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from a document.

Rules:
1. Answer ONLY using information found in the context below.
2. If the answer cannot be found in the context, respond exactly with:
   "Not found in document"
3. Do not use any outside knowledge or make assumptions.
4. Be concise and directly answer the question.
5. If helpful, you may quote short snippets from the context.

Context:
{context}

Question: {question}

Answer:"""
