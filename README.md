
# Mini RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using custom CRM documents and AI language models.

## Live Demo

🚀 **Public Application:** https://mini-rag-pipeline-1.onrender.com

## Project Overview

This project implements two versions of a RAG application:

- **Local Version:** Uses Ollama with Llama 3.2 to generate responses locally.
- **Public Version:** Uses Google Gemini API and is deployed on Render for public access.

## Features

- Loads documents from TXT and Markdown files.
- Splits documents into smaller text chunks.
- Uses TF-IDF and cosine similarity for document retrieval.
- Retrieves relevant CRM information.
- Supports CRM RAG and General AI modes.
- Generates answers using Llama 3.2 or Gemini.
- Displays retrieved document sources and similarity scores.
- Provides a Gradio web interface.

## Technologies Used

- Python
- Gradio
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Ollama
- Llama 3.2
- Google Gemini API
- Git and GitHub
- Render

## Project Structure

```text
mini-rag-pipeline/
│
├── data/
│   ├── duplicate_detection.txt
│   ├── lead_enrichment.txt
│   ├── webhooks_and_apis.txt
│   └── ai_email_workflow.txt
│
├── app.py
├── public_app.py
├── rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. User enters a question.
2. The application searches stored documents.
3. TF-IDF and cosine similarity identify relevant text chunks.
4. The best similarity score is compared with a threshold.
5. The application selects CRM RAG or General AI mode.
6. The selected LLM generates the response.
7. The answer and source documents are displayed.

## Run the Local Version

### 1. Clone the repository

```bash
git clone https://github.com/ggunasree068-hue/mini-rag-pipeline.git
cd mini-rag-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and download Llama 3.2

Install Ollama from https://ollama.com/download

```bash
ollama pull llama3.2
```

### 4. Start the application

```bash
python app.py
```

Open http://127.0.0.1:7860

## Run the Gemini Version

### 1. Get a Gemini API key

https://aistudio.google.com/app/apikey

### 2. Set the API key

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

### 3. Start the application

```bash
python public_app.py
```

Open http://127.0.0.1:7860

**Important:** Never upload your API key to GitHub. Configure it privately as an environment variable.

## Deployment

The public application is deployed using Render.

- Repository: GitHub
- Hosting: Render
- Build command: `pip install -r requirements.txt`
- Start command: `python public_app.py`
- API integration: Google Gemini

Live URL: https://mini-rag-pipeline-1.onrender.com

Note: Render's free service may sleep after inactivity. Gemini API usage is subject to applicable quotas and limits.

## Retrieval Configuration

- Chunk size: 60 words
- Chunk overlap: 15 words
- Top K: 3
- Similarity threshold: 0.13

The similarity score is a text-matching score, not an answer accuracy percentage.

## Example Questions

- How do we detect duplicate leads?
- How can we enrich lead information?
- What are webhooks and APIs?
- Explain Python loops with examples.

## Future Improvements

- Improve document retrieval accuracy.
- Implement semantic search using embeddings.
- Add more CRM documents.
- Improve the user interface.
- Add conversation history.

## Author

**Guna Sree**

GitHub: https://github.com/ggunasree068-hue
