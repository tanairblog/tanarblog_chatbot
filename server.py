import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
VECTOR_STORE_FILE = 'vector_store.pkl'
EMBEDDING_MODEL = 'models/text-embedding-004'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file.")

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
model = genai.GenerativeModel('gemini-flash-latest', generation_config=generation_config)

# Global variables for data
store = {}

import gzip

def load_resources():
    global store
    
    # Check for compressed file first
    compressed_file = VECTOR_STORE_FILE + '.gz'
    
    if os.path.exists(compressed_file):
        print(f"Loading vector store from {compressed_file}...")
        with gzip.open(compressed_file, 'rb') as f:
            store = pickle.load(f)
        print("Vector store loaded (Compressed).")
        print("Using Cloud Embeddings (0MB RAM usage).")
        
    elif os.path.exists(VECTOR_STORE_FILE):
        print(f"Loading vector store from {VECTOR_STORE_FILE}...")
        with open(VECTOR_STORE_FILE, 'rb') as f:
            store = pickle.load(f)
        print("Vector store loaded.")
        print("Using Cloud Embeddings (0MB RAM usage).")
    else:
        print("Vector store not found. Please run ingest.py first.")

def search(query, top_k=3):
    if not store:
        return []
    
    # Embed query using Gemini API
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(result['embedding'])
    except Exception as e:
        print(f"Embedding error: {e}")
        return []
    
    # Cosine similarity
    scores = np.dot(store['embeddings'], query_embedding)
    
    
    # Recency Boost
    timestamps = np.array([c.get('timestamp', 0) for c in store['chunks']])
    if len(timestamps) > 0:
        max_ts = np.max(timestamps)
        # Find min non-zero timestamp for better scaling range
        non_zero_ts = timestamps[timestamps > 0]
        min_ts = np.min(non_zero_ts) if len(non_zero_ts) > 0 else 0
        
        range_ts = max_ts - min_ts
        if range_ts > 0:
            # Min-Max scaling: (x - min) / (max - min)
            # This makes the oldest article 0.0 and newest 1.0
            normalized_ts = (timestamps - min_ts) / range_ts
            # 0 timestamps (missing) become negative, which is good (penalty)
        else:
            normalized_ts = np.zeros_like(timestamps, dtype=float)

        # Apply boost: Score + 10.0 * Recency
        # Extreme boost to ensure 2024 articles (Score ~14) always beat 2010 articles (Score ~5-7)
        # This effectively sorts search results by date, then by relevance.
        scores = scores + (10.0 * normalized_ts)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = store['chunks'][idx]
        res = chunk.copy()
        
        # Format year
        ts = res.get('timestamp', 0)
        if ts > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(ts / 1000)
            res['year'] = dt.year
        else:
            res['year'] = '????'
            
        results.append(res)
    return results

@app.route('/chat', methods=['POST'])
def chat():
    if not store:
        return jsonify({"error": "System not ready. Vector store missing."}), 500
        
    data = request.json
    user_query = data.get('message', '')
    history = data.get('history', [])
    
    if not user_query:
        return jsonify({"error": "Empty message"}), 400

    # 1. Query Reformulation (if history exists)
    search_query = user_query
    if history:
         try:
            # Create a simple conversation string for the rewriter
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
            rewrite_prompt = f"""Given the following conversation history, rewrite the last user query to be a standalone search query that contains all necessary context.
            
            Conversation:
            {history_str}
            User: {user_query}
            
            Standalone Query:"""
            
            rewritten = model.generate_content(rewrite_prompt).text.strip()
            print(f"Original Query: {user_query}")
            print(f"Rewritten Query: {rewritten}")
            search_query = rewritten
         except Exception as e:
            print(f"Rewriting failed: {e}")
            # Fallback to original
            pass

    # 2. Retrieve Context using Optimized Query
    relevant_chunks = search(search_query, top_k=5)
    
    context_str = "\n\n".join([f"Article ({r['year']}): {r['title']}\n{r['text']}" for r in relevant_chunks])
    
    # 3. Build Prompt with History
    history_context = ""
    if history:
        history_context = "Conversation History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]]) + "\n\n"

    prompt = f"""You are a helpful assistant for the website 'Tanarblog'.
    
    CRITICAL INSTRUCTION: You MUST answer in HUNGARIAN language.
    
    {history_context}
    
    Goal: Answer the user's latest question using the provided context.
    
    Instructions:
    - Answer in HUNGARIAN.
    - Use the provided articles to answer the question.
    - Prioritize information from newer articles (2020-2024) if available.
    - If the exact answer is not explicitly written, synthesize relevant advice.
    
    Context:
    {context_str}
    
    User Question: {user_query}
    """
    
    # 4. Generate Answer
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        answer = f"Sorry, I encountered an error communicating with the AI. Details: {str(e)}"

    # Return answer + sources
    return jsonify({
        "response": answer,
        "sources": [{"title": c['title'], "url": c['url'], "year": c['year']} for c in relevant_chunks]
    })

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Load resources immediately (for Gunicorn)
load_resources()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
