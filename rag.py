"""Small RAG (Retrieval-Augmented Generation) demo.

Pipeline: load documents -> split into chunks -> embed chunks -> retrieve the
most similar chunks for a question -> send them to an LLM as context.

Usage:
    python rag.py "How do we detect duplicate leads?"
    python rag.py            # interactive mode
"""
import os
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 60      # words per chunk
CHUNK_OVERLAP = 15   # words shared between neighbouring chunks
TOP_K = 3


def load_documents(folder=DATA_DIR):
    """Read every .txt / .md file in the folder -> list of (filename, text)."""
    docs = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word windows so ideas aren't cut in half."""
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
    """Chunk all documents and embed them. Returns (chunks, vectors)."""
    from sentence_transformers import SentenceTransformer  # lazy import

    chunks = []  # each: {"source": filename, "text": chunk}
    for name, text in docs:
        for piece in chunk_text(text):
            chunks.append({"source": name, "text": piece})
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    return model, chunks, np.asarray(vectors)


def retrieve(question, model, chunks, vectors, k=TOP_K):
    """Return the k chunks most similar to the question (cosine similarity)."""
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q  # vectors are normalised, so dot product = cosine
    best = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best]


def build_prompt(question, hits):
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in hits)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate(prompt):
    """Call an LLM if an API key is set; otherwise return None."""
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client()
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return client.models.generate_content(model=model, contents=prompt).text
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        msg = client.messages.create(
            model=model, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    return None


def answer(question, model, chunks, vectors):
    hits = retrieve(question, model, chunks, vectors)
    print("\nRetrieved context:")
    for c, score in hits:
        print(f"  ({score:.2f}) [{c['source']}] {c['text'][:90]}...")
    reply = generate(build_prompt(question, hits))
    if reply is None:
        print("\n(No API key set - showing retrieval only. "
              "Set GEMINI_API_KEY or ANTHROPIC_API_KEY to get a generated answer.)")
    else:
        print(f"\nAnswer: {reply}")


def main():
    docs = load_documents()
    if not docs:
        sys.exit(f"No .txt or .md files found in {DATA_DIR}")
    model, chunks, vectors = build_index(docs)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")

    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]), model, chunks, vectors)
        return
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        answer(q, model, chunks, vectors)


if __name__ == "__main__":
    main()
