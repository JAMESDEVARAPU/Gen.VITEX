import random
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import json
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key="YOURS API KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pickle"

chunks = json.loads(open(CHUNKS_FILE, 'rb').read())
def load_index(filename):
    return faiss.read_index(filename)

index = load_index(INDEX_FILE)

def get_similar(query, K=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, K)
    results = [(chunks[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return results

def generate_response(context, query):
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nResponse:"
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information based on the given context.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        max_tokens=150,
    )
    
    return chat_completion.choices[0].message.content.strip()

def handle_user_query(message, history):
    similar_chunks = get_similar(message, K=3)
    context = " ".join([chunk[0] for chunk in similar_chunks])
    
    response = generate_response(context, message)
    return response

gr.ChatInterface(handle_user_query, type="messages").launch()


