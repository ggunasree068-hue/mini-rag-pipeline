
# Mini RAG Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that answers questions using your own documents and also handles general questions using a locally running Large Language Model (LLM).

Built with **Python, TF-IDF, Gradio, Ollama, and Llama 3.2**.

## Features

- **Document Retrieval:** Searches relevant information from uploaded TXT and Markdown files.
- **CRM RAG Mode:** Answers CRM-related questions using retrieved document context.
- **General AI Mode:** Answers general questions using Llama 3.2 when relevant document context is not found.
- **Local LLM:** Runs Llama 3.2 through Ollama on your computer.
- **Web Interface:** Provides an interactive interface using Gradio.
- **Source Information:** Displays retrieved document names and similarity scores.

## How It Works

1. **Load Documents:** Reads `.txt` and `.md` files from the `data/` folder.
2. **Text Chunking:** Splits documents into overlapping 60-word chunks.
3. **Vectorization:** Converts document chunks into TF-IDF vectors.
4. **Retrieval:** Finds the top 3 most relevant document chunks using similarity scores.
5. **Mode Selection:** Selects CRM RAG or General AI based on the similarity threshold.
6. **Answer Generation:** Sends the prompt to Llama 3.2 through Ollama.
7. **Display:** Shows the generated answer in the Gradio interface.

## Architecture

```text
             User Question
                   |
                   v
              Gradio UI
                   |
                   v
            TF-IDF Retrieval
                   |
                   v
          Similarity Comparison
                   |
             +-----+-----+
             |           |
             v           v
          CRM RAG     General AI
             |           |
             v           v
       CRM Context   General Prompt
             |           |
             +-----+-----+
                   |
                   v
             Ollama API
                   |
                   v
              Llama 3.2
                   |
                   v
             Final Answer
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Gradio | Web interface |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorization and retrieval |
| Requests | HTTP communication |
| Ollama | Local LLM runtime |
| Llama 3.2 | Answer generation |

## Project Structure

```text
mini-rag-pipeline/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── ai_email_workflow.txt
    ├── duplicate_detection.txt
    ├── webhooks_and_apis.txt
    └── lead_enrichment.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ggunasree068-hue/mini-rag-pipeline.git

cd mini-rag-pipeline
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama:

https://ollama.com/download

### 4. Download Llama 3.2

Open PowerShell and run:

```bash
ollama pull llama3.2
```

### 5. Run the application

Make sure Ollama is running, then execute:

```bash
python app.py
```

Open the following URL in your browser:

```text
http://127.0.0.1:7860
```

## Example Questions

### CRM Questions

- How do we detect duplicate leads?
- Does the AI send emails automatically?
- What should a webhook receiver do?
- What happens to a lead with an invalid email?

### General Questions

- Explain Python functions with examples.
- What is Artificial Intelligence?
- Explain Python loops.
- What is Machine Learning?

## Configuration

The application uses the following default settings:

```python
CHUNK_SIZE = 60
CHUNK_OVERLAP = 15
TOP_K = 3
SIMILARITY_THRESHOLD = 0.13
OLLAMA_MODEL = "llama3.2"
```

These settings can be adjusted in `rag.py`.

## Run the Terminal Version

You can also use the RAG pipeline directly from the terminal:

```bash
python rag.py
```

Enter questions interactively and type `quit` to exit.

## Future Improvements

- Add PDF document support.
- Improve document retrieval accuracy.
- Integrate a vector database such as FAISS or Chroma.
- Add conversation history.
- Deploy a public web application.
- Add document upload functionality.

## Author

**Gangireddy Guna Sree**

GitHub: https://github.com/ggunasree068-hue
