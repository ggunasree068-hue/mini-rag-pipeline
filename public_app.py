
import os
import gradio as gr
from google import genai
import rag


# Gemini configuration
GEMINI_MODEL = "gemini-3.8-flash"

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# Load documents and build the RAG index
docs = rag.load_documents()

if not docs:
    raise ValueError("No documents found in the data folder.")

vectorizer, chunks, vectors = rag.build_index(docs)


# Generate answers using Gemini
def answer_question(question):

    if not question or not question.strip():
        return "Please enter a question."

    if client is None:
        return "Gemini API key is missing. Please configure GEMINI_API_KEY."

    # Retrieve relevant documents
    hits = rag.retrieve(
        question,
        vectorizer,
        chunks,
        vectors
    )

    best_score = max(
        (score for chunk, score in hits),
        default=0.0
    )

    # Select RAG or General AI mode
    if best_score >= rag.SIMILARITY_THRESHOLD:

        mode = "CRM RAG"
        prompt = rag.build_prompt(question, hits)

        sources = "\n".join(
            f"- {chunk['source']} (similarity: {score:.2f})"
            for chunk, score in hits
        )

    else:

        mode = "General AI"
        prompt = rag.build_general_prompt(question)

        sources = "No relevant CRM documents found."

    # Ask Gemini
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        reply = response.text

        if not reply:
            return "Gemini did not generate a response. Please try again."

    except Exception as error:
        return f"Gemini API error: {error}"

    return (
        f"Mode: {mode}\n"
        f"Best similarity score: {best_score:.2f}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Answer:\n{reply}"
    )


# Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="Enter your question here...",
        lines=2
    ),
    outputs=gr.Textbox(
        label="Gemini RAG Answer",
        lines=15
    ),
    title="Mini RAG Assistant - Public Version",
    description=(
        "Ask questions about CRM documents "
        "or ask general questions using Google Gemini."
    )
)


# Launch application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
