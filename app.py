"""
app.py
------
Bonus: Simple Streamlit UI for the RAG Chatbot.

Run with:
    streamlit run app.py

Flow:
  1. Upload a PDF in the sidebar (or it auto-detects an existing index).
  2. Click "Build Index" to run Load -> Split -> Embed -> Store.
  3. Ask questions in the chat box; answers are generated only from
     the document, with sources shown for verification.
"""

import os
import tempfile
import streamlit as st

from config import VECTORSTORE_DIR, GOOGLE_API_KEY
from ingest import load_pdf, split_documents
from vectorstore_utils import build_vectorstore, load_vectorstore
from qa_chain import build_qa_chain, ask


st.set_page_config(page_title="RAG Chatbot (LangChain)", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot — Ask Your Document")
st.caption(
    "Built with LangChain + FAISS (local) + HuggingFace embeddings + Gemini. "
    "Answers ONLY from the uploaded document — says 'Not found in document' otherwise."
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = os.path.exists(VECTORSTORE_DIR) and bool(
        os.listdir(VECTORSTORE_DIR)
    )


# --- Sidebar: setup ---
with st.sidebar:
    st.header("⚙️ Setup")

    if not GOOGLE_API_KEY:
        st.warning(
            "⚠️ No GOOGLE_API_KEY found.\n\n"
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to a `.env` file as:\n\n`GOOGLE_API_KEY=your_key_here`"
        )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🔧 Build Index (Load → Split → Embed → Store)", type="primary"):
            with st.spinner("Processing PDF..."):
                # Save uploaded file to a temp path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Step 2: Load
                docs = load_pdf(tmp_path)
                # Step 3: Split
                chunks = split_documents(docs)
                # Step 4: Embed & Store
                build_vectorstore(chunks)

                os.remove(tmp_path)

            st.session_state.index_ready = True
            st.session_state.qa_chain = None  # force rebuild
            st.session_state.messages = []
            st.success(f"Index built from {len(chunks)} chunks! Ask away below.")

    st.divider()

    if st.session_state.index_ready:
        st.success("✅ Vector index ready")
    else:
        st.info("ℹ️ Upload a PDF and build the index to start.")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# --- Load QA chain (lazy, cached in session) ---
def get_chain():
    if st.session_state.qa_chain is None:
        with st.spinner("Loading vector store and model..."):
            vectorstore = load_vectorstore()
            st.session_state.qa_chain = build_qa_chain(vectorstore)
    return st.session_state.qa_chain


# --- Main chat area ---
if not st.session_state.index_ready:
    st.info("👈 Upload a PDF and click **Build Index** in the sidebar to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] Page {src['page']}**")
                        st.text(src["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not GOOGLE_API_KEY:
                answer = (
                    "⚠️ GOOGLE_API_KEY is not configured. Please add a free Gemini "
                    "API key to your `.env` file (see README for instructions)."
                )
                sources_data = []
                st.markdown(answer)
            else:
                with st.spinner("Thinking..."):
                    try:
                        chain = get_chain()
                        answer, sources = ask(chain, question)
                        sources_data = [
                            {
                                "page": doc.metadata.get("page", "?"),
                                "content": doc.page_content[:500],
                            }
                            for doc in sources
                        ]
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                        sources_data = []

                st.markdown(answer)
                if sources_data:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources_data, 1):
                            st.markdown(f"**[{i}] Page {src['page']}**")
                            st.text(src["content"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources_data}
        )
