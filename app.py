
import os
import gradio as gr
import rag


# Load documents and build the index once
docs = rag.load_documents()

if not docs:
    raise ValueError("No documents found in the data folder.")

vectorizer, chunks, vectors = rag.build_index(docs)


# Answer questions using CRM RAG or General AI
def answer_question(question):

    if not question or not question.strip():
        return "Please enter a question."

    # Retrieve relevant document chunks
    hits = rag.retrieve(
        question,
        vectorizer,
        chunks,
        vectors
    )

    # Find the highest similarity score
    best_score = max(
        (score for chunk, score in hits),
        default=0.0
    )

    # Select the appropriate mode
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

    # Generate answer using Ollama
    reply = rag.generate(prompt)

    if reply is None:
        return (
            "Unable to connect to Ollama.\n"
            "Please make sure Ollama is running."
        )

    # Display answer and retrieval information
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
        label="RAG Answer",
        lines=15
    ),
    title="Mini RAG Assistant",
    description=(
        "Ask questions about your CRM documents "
        "or ask general questions using Llama 3.2."
    )
)


# Launch application
demo.launch(
    server_name="127.0.0.1",
    server_port=7860
)
