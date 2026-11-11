# Mini RAG Demo

A small Retrieval-Augmented Generation (RAG) pipeline in plain Python. It answers questions using only the text files in the `data` folder, so the LLM answers from your documents instead of guessing.

## How it works

1. **Load** the .txt files in `data`.
2. **Chunk** each document into overlapping 60-word pieces.
3. **Embed** every chunk with sentence-transformers (all-MiniLM-L6-v2).
4. **Retrieve** the 3 chunks most similar to the question (cosine similarity).
5. **Generate** an answer by giving those chunks to an LLM as context. The prompt tells it to answer only from that context.

## Run it

pip install -r requirements.txt
python rag.py "How do we detect duplicate leads?"

Optional: set GEMINI_API_KEY or ANTHROPIC_API_KEY to get a generated answer. Without a key, the program shows the retrieved text only.

## Example questions

- How do we detect duplicate leads?
- Does the AI send emails automatically?
- What should a webhook receiver do?
- What happens to a lead with an invalid email?

## Add your own documents

Put any .txt file in the `data` folder and run the program again.# mini-rag-pipeline
Small RAG pipeline: chunking, embeddings, retrieval, LLM answers.
