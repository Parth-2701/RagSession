"""
test_chatbot.py
----------------
Step 6: Ask 5 sample questions against the indexed document and print
the answers + sources, so you can manually verify the answers are
grounded in the document (not hallucinated).

Usage:
    python test_chatbot.py
    python test_chatbot.py "Custom question 1" "Custom question 2"
"""

import sys
from qa_chain import build_qa_chain, ask

DEFAULT_QUESTIONS = [
    "What is this document about?",
    "Summarize the main points in a few sentences.",
    "What are the key requirements or steps mentioned?",
    "Are there any specific tools, technologies, or names mentioned?",
    "What is the capital of France?",  # intentionally off-topic, should say "Not found in document"
]


def main():
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    print("Building QA chain (loading FAISS index + Gemini)...\n")
    chain = build_qa_chain()

    for i, q in enumerate(questions, 1):
        print(f"{'='*60}")
        print(f"Q{i}: {q}")
        print(f"{'='*60}")
        answer, sources = ask(chain, q)
        print(f"Answer: {answer}\n")
        print("Sources used:")
        if sources:
            for j, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content[:120].replace("\n", " ")
                print(f"  [{j}] page {page}: {snippet}...")
        else:
            print("  (none)")
        print()


if __name__ == "__main__":
    main()
