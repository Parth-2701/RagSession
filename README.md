# 📄 RAG Chatbot with LangChain

A document-grounded Q&A chatbot built with **LangChain**, **FAISS**, free **HuggingFace embeddings**,
and **Google Gemini** (free tier) for generation. Answers questions **only** using content from
the uploaded PDF — if the answer isn't in the document, it replies `"Not found in document"`.

---

## 🏗️ Architecture

```
PDF File
   │
   ▼
[1] PyPDFLoader            → Load PDF into LangChain Documents (per page)
   │
   ▼
[2] RecursiveCharacterTextSplitter → Split into ~1000-char chunks (200 overlap)
   │
   ▼
[3] HuggingFaceEmbeddings (all-MiniLM-L6-v2) → Convert chunks to vectors (runs locally, free)
   │
   ▼
[4] FAISS Vector Store     → Persisted to disk (vectorstore/)
   │
   ▼
User Query
   │
   ▼
[5] Retriever (top-k=4)    → Finds most relevant chunks via similarity search
   │
   ▼
[6] RetrievalQA Chain       → Stuffs chunks + strict prompt into Gemini (gemini-1.5-flash)
   │
   ▼
Answer (grounded in document, or "Not found in document")
   │
   ▼
[Bonus] Streamlit UI         → Chat interface with source citations
```

---

## 📂 Project Structure

```
rag_chatbot/
├── app.py                # Streamlit UI (bonus)
├── config.py             # Central settings (chunk size, models, prompt)
├── ingest.py             # Step 1-2: Load PDF + split into chunks
├── vectorstore_utils.py  # Step 3-4: Embeddings + FAISS build/load
├── qa_chain.py            # Step 5-6: RetrievalQA chain (Gemini)
├── test_chatbot.py        # Step 6: Ask 5 test questions, check grounding
├── requirements.txt
├── .env.example
├── data/                  # (optional) put your PDFs here
└── vectorstore/           # Generated FAISS index (created after ingestion)
```

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FREE Gemini API key
- Go to https://aistudio.google.com/app/apikey
- Click "Create API key" (free tier, no credit card required)
- Copy the key

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and paste your key:
```
GOOGLE_API_KEY=your_actual_key_here
```

---

## 🧪 Usage

### Option A — Command line

**Build the index from your PDF:**
```bash
python vectorstore_utils.py data/your_document.pdf
```

**Ask questions:**
```bash
python qa_chain.py "What is this document about?"
```

Or run interactively:
```bash
python qa_chain.py
```

**Run the 5-question grounding test:**
```bash
python test_chatbot.py
```

### Option B — Streamlit UI (recommended / bonus)

```bash
streamlit run app.py
```

1. Upload your PDF in the sidebar
2. Click **"Build Index"**
3. Ask questions in the chat box — see source chunks for each answer

---

## 🧠 How "No Hallucination" Is Enforced

The prompt in `config.py` (`QA_PROMPT_TEMPLATE`) explicitly instructs the LLM:

- Use **only** the retrieved context
- Reply exactly `"Not found in document"` if the answer isn't present
- No outside knowledge or assumptions

Combined with `temperature=0.0`, this keeps answers deterministic and grounded.

---

## 🆓 Free-Tier Notes

| Component   | Choice                              | Cost |
|-------------|-------------------------------------|------|
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` | Free, runs locally (CPU) |
| Vector Store| FAISS                                | Free, local |
| LLM         | Google Gemini `gemini-1.5-flash`     | Free tier via API key |

No OpenAI credits required.

---

## 🔧 Customization

Edit `config.py` to tune:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking granularity
- `TOP_K` — number of retrieved chunks per query
- `EMBEDDING_MODEL_NAME` — swap embedding model
- `GEMINI_MODEL_NAME` — swap Gemini model variant
- `QA_PROMPT_TEMPLATE` — adjust grounding/prompt behavior

---
