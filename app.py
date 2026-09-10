
import os
import pickle
import faiss
import gradio as gr
from groq import Groq

print("Loading pre-built index...")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load chunks and metadata
with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
YOUR_NAME = data["your_name"]

print(f"✅ Loaded {len(chunks)} chunks and index")

# Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Lazy load model
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        print("First query - loading tiny model...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        print("✅ Model loaded")
    return embedder

def retrieve(query, k=4):
    import numpy as np
    emb = get_embedder()
    q_vec = emb.encode([query]).astype("float32")
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
        f"mentioned in the excerpts. Never say \"probably\" or \"might have\" - only state facts.\n\n"

        f"4. Always refer to {YOUR_NAME} in third person (e.g., \"{YOUR_NAME} has experience in...\").\n\n"

        f"5. Keep responses professional, friendly, and concise (2-4 sentences unless asked for details).\n\n"

        f"Information excerpts:\n{context}\n\n"

        f"Remember: Be helpful and honest. If you don\'t know, say so."
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
    # Gradio history: list of [user_msg, bot_msg] pairs
    chat_history = []
    for user_msg, bot_msg in history:
        chat_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            chat_history.append({"role": "assistant", "content": bot_msg})
    return ask_personal_chatbot(message, chat_history=chat_history)

def health():
    return {"status": "ok"}

# Minimal ChatInterface (no theme, no type - maximum compatibility)
demo = gr.ChatInterface(
    fn=gradio_chat_fn,
    title=f"Chat with {YOUR_NAME}\'s AI Assistant",
    description="Ask me about my background, skills, and projects!"
)

demo.app.get("/health")(health)

if __name__ == "__main__":
    print("Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
