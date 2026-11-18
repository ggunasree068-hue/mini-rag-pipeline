
"""Mini RAG pipeline using TF-IDF and Ollama Llama 3.2."""

import sys
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_DIR = Path(__file__).parent / "data"

CHUNK_SIZE = 60
CHUNK_OVERLAP = 15
TOP_K = 3

# Adjust this threshold after testing your documents.
SIMILARITY_THRESHOLD = 0.13

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def load_documents(folder=DATA_DIR):
    """Load all text and Markdown documents."""

    docs = []

    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append(
                (path.name, path.read_text(encoding="utf-8"))
            )

    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks."""

    words = text.split()
    step = max(size - overlap, 1)

    chunks = []
    start = 0

    while start < len(words):
        piece = words[start:start + size]

        if piece:
            chunks.append(" ".join(piece))

        if start + size >= len(words):
            start = len(words)
        else:
            start += step

    return chunks


def build_index(docs):
    """Create TF-IDF vectors from document chunks."""

    chunks = []

    for name, text in docs:
        for piece in chunk_text(text):
            chunks.append({
                "source": name,
                "text": piece
            })

    if not chunks:
        raise ValueError("No text chunks found.")

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, chunks, vectors


def retrieve(question, vectorizer, chunks, vectors, k=TOP_K):
    """Retrieve the most relevant document chunks."""

    question_vector = vectorizer.transform([question])

    scores = (
        vectors @ question_vector.T
    ).toarray().flatten()

    best = np.argsort(scores)[::-1][:k]

    return [
        (chunks[i], float(scores[i]))
        for i in best
    ]


def build_prompt(question, hits):
    """Build a prompt using retrieved CRM context."""

    context = "\n\n".join(
        f"[{chunk['source']}] {chunk['text']}"
        for chunk, score in hits
    )

    return (
        "You are a helpful CRM assistant. "
        "Answer using only the provided CRM context. "
        "Do not invent CRM details. "
        "If the information is missing, say so.\n\n"
        f"CRM Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_general_prompt(question):
    """Build a prompt for general questions."""

    return (
        "You are a helpful general-purpose AI assistant. "
        "Answer the user's question clearly and accurately. "
        "Use simple language and examples when helpful.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate(prompt):
    """Generate an answer using Llama 3.2 through Ollama."""

    try:
        print(
            "\nConnecting to Ollama (Llama 3.2)...",
            flush=True
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()
        answer_text = result.get("response", "").strip()

        if not answer_text:
            print("Ollama returned an empty response.")
            return None

        print("Llama response received!")

        return answer_text

    except requests.exceptions.RequestException as error:
        print(f"\nOllama connection error: {error}")
        return None

    except (ValueError, KeyError) as error:
        print(f"\nOllama response error: {error}")
        return None


def answer(question, vectorizer, chunks, vectors):
    """Answer CRM and general questions using two modes."""

    hits = retrieve(
        question,
        vectorizer,
        chunks,
        vectors
    )

    best_score = max(
        (score for chunk, score in hits),
        default=0.0
    )

    print(f"\nBest similarity score: {best_score:.2f}")

    if best_score >= SIMILARITY_THRESHOLD:

        print("\nMode: CRM RAG")

        print("\nRetrieved context:")

        for chunk, score in hits:
            print(
                f"  ({score:.2f}) "
                f"[{chunk['source']}] "
                f"{chunk['text'][:90]}..."
            )

        prompt = build_prompt(question, hits)

    else:

        print("\nMode: General AI")

        print(
            "No sufficiently relevant CRM context found. "
            "Using Llama's general knowledge."
        )

        prompt = build_general_prompt(question)

    reply = generate(prompt)

    if reply is None:
        print("\nUnable to generate a response. Check Ollama.")
    else:
        print(f"\nAnswer:\n{reply}")


def main():
    """Run the complete RAG pipeline."""

    docs = load_documents()

    if not docs:
        sys.exit(
            f"No documents found in {DATA_DIR}"
        )

    vectorizer, chunks, vectors = build_index(docs)

    print(
        f"Indexed {len(chunks)} chunks "
        f"from {len(docs)} documents."
    )

    if sys.argv[1:]:

        question = " ".join(sys.argv[1:])

        answer(
            question,
            vectorizer,
            chunks,
            vectors
        )

        return

    question = input(
        "\nAsk a question (or 'quit'): "
    ).strip()

    while question.lower() not in {"quit", "exit", ""}:

        answer(
            question,
            vectorizer,
            chunks,
            vectors
        )

        question = input(
            "\nAsk a question (or 'quit'): "
        ).strip()


if __name__ == "__main__":
    main()
