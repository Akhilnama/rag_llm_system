# src/pipeline.py

from src.ingestion import load_pdf
from src.chunking import chunk_text
from src.embedding import create_embeddings, build_faiss_index, model
from src.retrieval import retrieve
from src.llm import LLMRouter
import os
from dotenv import load_dotenv
from pathlib import Path

# Force load from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ----------------------------
# CONFIG
# ----------------------------``
CONFIG = {
    "groq": os.getenv("GROQ_API_KEY")
}

if CONFIG["groq"] is None:
    raise ValueError("GROQ_API_KEY not found. Check your .env setup.")

# ----------------------------
# INITIALISE SYSTEM
# ----------------------------
print("Loading documents...")
text = load_pdf(rf"{Path(__file__).resolve().parent.parent}\data\raw\sample.pdf")

print("Chunking...")
chunks = chunk_text(text)

print("Creating embeddings...")
embeddings = create_embeddings(chunks)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

print("Initialising LLM...")
llm = LLMRouter(CONFIG)


# ----------------------------
# CORE QUERY FUNCTION
# ----------------------------
def run_query(query: str, provider: str = "groq"):
    # Step 1: Retrieve context
    context = retrieve(query, model, index, chunks)

    # Step 2: Build context text
    context_text = "\n".join([c["text"] for c in context])

    # Step 3: Prompt
    prompt = f"""
You are a financial assistant.

Answer ONLY using the provided context.
If answer is not in context, say "I don't know".

Context:
{context_text}

Question:
{query}

Answer:
"""

    # Step 4: Generate response
    answer = llm.generate(prompt, provider=provider)

    return {
        "query": query,
        "context": context,
        "answer": answer
    }


# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":
    test_query = "what is mentioned in the document"
    result = run_query(test_query)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== ANSWER ===")
    print(result["answer"])