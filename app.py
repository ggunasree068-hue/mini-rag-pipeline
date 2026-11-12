import os
import gradio as gr
import rag

# Load documents and build the index once
docs = rag.load_documents()

if not docs:
    raise ValueError("No documents found in the data folder.")

model, chunks, vectors = rag.build_index(docs)


# Function to answer user questions
def answer_question(question):

    if not question.strip():
        return "Please enter a question."

    # Retrieve relevant document chunks
    hits = rag.retrieve(question, model, chunks, vectors)

    if hits[0][1] < 0.30:
        return "Sorry, I don't have information about this topic in my documents."
    # Generate answer using retrieved context
    prompt = rag.build_prompt(question, hits)
    reply = rag.generate(prompt)

    if reply is None:
        result = "No API key found. Showing retrieved context:\n\n"

        for chunk, score in hits:
            result += (
                f"Source: {chunk['source']}\n"
                f"Similarity: {score:.2f}\n"
                f"{chunk['text']}\n\n"
            )

        return result

    return reply


# Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="Enter your question here..."
    ),
    outputs=gr.Textbox(label="RAG Answer"),
    title="Mini RAG Assistant",
    description="Ask questions about your uploaded documents."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)