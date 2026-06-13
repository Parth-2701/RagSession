"""
qa_chain.py
-----------
Step 5 of the pipeline: Wire the FAISS retriever into a RetrievalQA chain
backed by Google Gemini's free-tier API.

The prompt strictly instructs the LLM to answer ONLY from retrieved context
and to say "Not found in document" otherwise (anti-hallucination guardrail).
"""

import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    TOP_K,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    QA_PROMPT_TEMPLATE,
)
from vectorstore_utils import load_vectorstore


def get_llm():
    """
    Returns the Gemini chat model (free tier).
    Raises a clear error if no API key is configured.
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/app/apikey and add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,  # deterministic, grounded answers
        convert_system_message_to_human=True,
    )


def build_qa_chain(vectorstore=None):
    """
    Step 5: Build the full RetrievalQA chain.
      - Retriever: FAISS similarity search (top-k)
      - LLM: Gemini (temperature=0 for grounded, factual answers)
      - Prompt: strict, anti-hallucination instructions
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def ask(qa_chain, question: str):
    """
    Step 6 helper: Ask a question and return (answer, source_documents).
    """
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources


if __name__ == "__main__":
    import sys

    chain = build_qa_chain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, sources = ask(chain, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}\n")
        print("--- Sources ---")
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get("page", "?")
            print(f"[{i}] (page {page}) {doc.page_content[:150]}...")
    else:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            q = input("Question: ")
            if q.lower() in ("exit", "quit"):
                break
            answer, sources = ask(chain, q)
            print(f"Answer: {answer}\n")
