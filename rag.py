"""Small RAG (Retrieval-Augmented Generation) demo.

Pipeline: load documents -> split into chunks -> embed chunks -> retrieve the
most similar chunks for a question -> send them to an LLM as context.

Usage:
    python rag.py "How do we detect duplicate leads?"
    python rag.py            # interactive mode
"""

"""Mini RAG pipeline using TF-IDF and Gemini."""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Folder containing text documents
DATA_DIR = Path(__file__).parent / "data"

# RAG settings
CHUNK_SIZE = 60
CHUNK_OVERLAP = 15
TOP_K = 3


def load_documents(folder=DATA_DIR):
    """Load all .txt and .md files from the data folder."""

    docs = []

    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))

    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split document text into smaller overlapping chunks."""

    words = text.split()
    step = max(size - overlap, 1)

    chunks = []

    for start in range(0, len(words), step):
        piece = words[start:start + size]

        if piece:
            chunks.append(" ".join(piece))

        if start + size >= len(words):
            break

    return chunks


def build_index(docs):
    """Convert document chunks into TF-IDF vectors."""

    chunks = []

    for name, text in docs:
        for piece in chunk_text(text):
            chunks.append({
                "source": name,
                "text": piece
            })

    if not chunks:
        raise ValueError("No text chunks found in the documents.")

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [c["text"] for c in chunks]
    )

    return vectorizer, chunks, vectors


def retrieve(question, vectorizer, chunks, vectors, k=TOP_K):
    """Find the most relevant document chunks using TF-IDF."""

    q = vectorizer.transform([question])

    scores = (vectors @ q.T).toarray().flatten()

    best = np.argsort(scores)[::-1][:k]

    return [(chunks[i], float(scores[i])) for i in best]


def build_prompt(question, hits):
    """Create a prompt using retrieved document chunks."""

    context = "\n\n".join(
        f"[{c['source']}] {c['text']}" for c, _ in hits
    )

    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )



def generate(prompt):
    """Generate an answer using Gemini or Claude."""

    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai

            print("\nConnecting to Gemini...")

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

            model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            print("Gemini response received.")

            if response.text:
                return response.text

            return "Gemini returned an empty response."

        except Exception as e:
            print(f"\nGemini error: {e}")
            return None

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

            msg = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            return msg.content[0].text

        except Exception as e:
            print(f"\nClaude error: {e}")
            return None

    return None



def answer(question, vectorizer, chunks, vectors):
    """Retrieve relevant chunks and generate an answer."""

    hits = retrieve(question, vectorizer, chunks, vectors)

    print("\nRetrieved context:")

    for c, score in hits:
        print(
            f"  ({score:.2f}) [{c['source']}] "
            f"{c['text'][:90]}..."
        )

    reply = generate(build_prompt(question, hits))

    if reply is None:
        print(
            "\nNo API key set. Showing retrieval only. "
            "Set GEMINI_API_KEY or ANTHROPIC_API_KEY "
            "to get a generated answer."
        )
    else:
        print(f"\nAnswer: {reply}")


def main():
    """Load documents, build index and accept questions."""

    docs = load_documents()

    if not docs:
        sys.exit(f"No .txt or .md files found in {DATA_DIR}")

    vectorizer, chunks, vectors = build_index(docs)

    print(
        f"Indexed {len(chunks)} chunks "
        f"from {len(docs)} documents."
    )

    if len(sys.argv) > 1:
        answer(
            " ".join(sys.argv[1:]),
            vectorizer,
            chunks,
            vectors
        )
        return

    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()

        if q.lower() in {"quit", "exit", ""}:
            break

        answer(q, vectorizer, chunks, vectors)


if __name__ == "__main__":
    main()
