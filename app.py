
import os
import pickle
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Load your data (created in the notebook) ---
with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
YOUR_NAME = data["your_name"]

# --- Set up embedding model + index ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings).astype("float32"))

# --- Groq client (reads API key from environment variable) ---
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def retrieve(query, k=4):
    q_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]

def ask_personal_chatbot(query, chat_history=None, k=4):
    retrieved = retrieve(query, k=k)
    context = "\n\n---\n\n".join(retrieved)
    system_prompt = (
        f"You are {YOUR_NAME}\'s personal AI assistant, answering questions about "
        f"{YOUR_NAME}\'s background, skills, and projects on their behalf. "
        f"Answer ONLY using the resume/project excerpts provided below. "
        f"If something isn\'t covered, say you don\'t have that information yet "
        f"— never invent details. Keep answers friendly, concise, and in third person."
        f"\n\nExcerpts:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages, temperature=0.3,
        reasoning_format="hidden",
    )
    return response.choices[0].message.content

def gradio_chat_fn(message, history):
    chat_history = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            chat_history.append({"role": turn["role"], "content": turn["content"]})
    return ask_personal_chatbot(message, chat_history=chat_history)

# Health endpoint for UptimeRobot (doesn\'t call Groq API!)
def health():
    return {"status": "ok"}

demo = gr.ChatInterface(
    fn=gradio_chat_fn,
    title=f"Chat with {YOUR_NAME}\'s AI Assistant",
    description="Ask me about my background, skills, and projects!",
    type="messages",
    theme=gr.themes.Soft(),  # Professional theme for portfolio embedding
)

# Mount health endpoint
demo.app.get("/health")(health)

if __name__ == "__main__":
    # Render needs server to bind to 0.0.0.0 and use PORT environment variable
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
