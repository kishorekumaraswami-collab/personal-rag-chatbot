
import os
import pickle
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

print("Loading pre-built index...")

# Load FAISS index (pre-built!)
index = faiss.read_index("faiss_index.bin")

# Load chunks and metadata
with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
YOUR_NAME = data["your_name"]

print(f"✅ Loaded {len(chunks)} chunks")

# Use TINY model (61MB only!)
print("Loading tiny embedding model...")
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
print("✅ Model loaded (61MB)")

# Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def retrieve(query, k=4):
    import numpy as np
    q_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]

def ask_personal_chatbot(query, chat_history=None, k=4):
    retrieved = retrieve(query, k=k)
    context = "\n\n---\n\n".join(retrieved)

    system_prompt = (
        f"You are {YOUR_NAME}\'s personal AI assistant. Your role is to help visitors "
        f"learn about {YOUR_NAME}\'s professional background, skills, and projects.\n\n"

        f"CRITICAL RULES:\n"
        f"1. If user greets (hi/hello/hey), respond: \"Hello! I\'m {YOUR_NAME}\'s AI assistant. "
        f"I can answer questions about {YOUR_NAME}\'s background, skills, projects, and experience. "
        f"What would you like to know?\"\n\n"

        f"2. Answer ONLY using information in the excerpts below. If the answer isn\'t in the excerpts, "
        f"respond: \"I don\'t have that specific information in {YOUR_NAME}\'s profile yet. You can reach "
        f"out to {YOUR_NAME} directly for more details.\"\n\n"

        f"3. NEVER fabricate or guess projects, skills, job titles, dates, or any details not explicitly "
        f"mentioned in the excerpts. Never say \"probably\" or \"might have\" - only state facts from the excerpts.\n\n"

        f"4. Always refer to {YOUR_NAME} in third person (e.g., \"{YOUR_NAME} has experience in...\", "
        f"\"{YOUR_NAME} worked on...\"). Never use first person.\n\n"

        f"5. Keep responses professional, friendly, and concise (2-4 sentences unless asked for details). "
        f"You\'re representing {YOUR_NAME} to recruiters and potential collaborators.\n\n"

        f"Information excerpts:\n{context}\n\n"

        f"Remember: Be helpful and honest. If you don\'t know, say so. Never overpromise or exaggerate."
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

def health():
    return {"status": "ok"}

demo = gr.ChatInterface(
    fn=gradio_chat_fn,
    title=f"Chat with {YOUR_NAME}\'s AI Assistant",
    description="Ask me about my background, skills, and projects!",
    type="messages",
    theme=gr.themes.Soft(),
)

demo.app.get("/health")(health)

if __name__ == "__main__":
    print("Starting Gradio on port 10000...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
