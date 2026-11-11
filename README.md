# Mini RAG Demo

A small Retrieval-Augmented Generation (RAG) pipeline in plain Python. It answers questions using only the text files in `data/`, so the LLM answers from your documents instead of guessing.

## How it works

1. **Load** the `.txt` / `.md` files in `data/`.
2. **Chunk** each document into overlapping 60-word pieces.
3. **Embed** every chunk with `sentence-transformers` (`all-MiniLM-L6-v2`).
4. **Retrieve** the top 3 chunks most similar to the question (cosine similarity).
5. **Generate** an answer by giving those chunks to an LLM as context. The prompt tells it to answer only from that context.

## Run it

```bash
pip install -r requirements.txt

# Optional: choose ONE LLM provider (without a key you still see retrieval results)
export GEMINI_API_KEY=your_key        # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
# or
export ANTHROPIC_API_KEY=your_key

python rag.py "How do we detect duplicate leads?"
python rag.py                          # interactive mode
```

  ![Example output](Screenshot%202026-11-11%20170549.png)

Model names can be changed with `GEMINI_MODEL` or `ANTHROPIC_MODEL` if a default is retired.

## Example questions

- How do we detect duplicate leads?
- Does the AI send emails automatically?
- What should a webhook receiver do?
- What happens to a lead with an invalid email?

## Add your own documents

Drop any `.txt` or `.md` file into `data/` and run again.

## Possible improvements

- Read PDFs
- Store vectors in FAISS or Chroma
- Show source citations in the answer
- Add a Streamlit interface
