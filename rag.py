import pdfplumber
import re
import unicodedata
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import sqlite3
from fastapi import FastAPI
import uuid
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = FastAPI()

# Global variables (initialized in main.py)
index = None
chunks = None

def extract_and_clean_text(pdf_path):
    """Extract text from a PDF and clean it for processing."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        text = re.sub(r'\s+', ' ', text.strip())
        text = unicodedata.normalize('NFKC', text)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def chunk_text(text, max_length=200):
    """Split text into chunks for vectorization."""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def vectorize_and_store(chunks, index_path='vector_index.index'):
    """Convert chunks to embeddings and store in FAISS."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index, chunks

def retrieve_chunks(query, index, chunks, k=3):
    """Retrieve top-k relevant chunks for a query."""
    query_embedding = model.encode([query])[0]
    faiss.normalize_L2(np.array([query_embedding]))
    distances, indices = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in indices[0]], distances[0]

generator = pipeline('text-generation', model='distilgpt2')

def generate_answer(query, chunks):
    """Generate an answer based on retrieved chunks."""
    context = " ".join(chunks)
    prompt = f"Query: {query}\nContext: {context}\nAnswer in one word or short phrase: "
    result = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
    answer = result[0]['generated_text'].split('\n')[-1].strip()
    return answer

def init_chat_history(db_path='chat_history.db'):
    """Initialize SQLite database for short-term memory."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id TEXT PRIMARY KEY, query TEXT, answer TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def save_chat(query, answer, db_path='chat_history.db'):
    """Save query and answer to chat history."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    chat_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO chat_history (id, query, answer, timestamp) VALUES (?, ?, ?, datetime('now'))",
                  (chat_id, query, answer))
    conn.commit()
    conn.close()

def get_recent_chats(limit=5, db_path='chat_history.db'):
    """Retrieve recent chat history."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT query, answer FROM chat_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    chats = cursor.fetchall()
    conn.close()
    return chats

def evaluate_retrieval(query, retrieved_chunks, expected_answer):
    """Evaluate retrieval and answer quality."""
    query_embedding = model.encode([query])[0]
    chunk_embeddings = model.encode(retrieved_chunks)
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    groundedness = any(expected_answer.lower() in chunk.lower() for chunk in retrieved_chunks)
    return {
        "similarity_scores": similarities.tolist(),
        "groundedness": groundedness
    }

@app.post("/query")
async def query_rag(data: dict):
    """API endpoint to process queries and return answers."""
    global index, chunks
    query = data.get("query", "")
    if not query:
        return {"error": "Query is required"}
    if index is None or chunks is None:
        return {"error": "System not initialized"}
    retrieved_chunks, distances = retrieve_chunks(query, index, chunks)
    answer = generate_answer(query, retrieved_chunks)
    save_chat(query, answer)
    evaluation = evaluate_retrieval(query, retrieved_chunks, answer)
    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "similarity_scores": distances.tolist(),
        "evaluation": evaluation
    }