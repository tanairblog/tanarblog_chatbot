import json
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration
JSON_FILE = 'cikkek.json'
VECTOR_STORE_FILE = 'vector_store.pkl'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # Better for Hungarian

def list_articles(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f) # The file contains line-delimited JSON objects? Or a list?
        # Based on view_file output, it looks like line-delimited JSON or a list of objects.
        # Let's check the first line in view_file. It was line 1: {...} line 2: {...}.
        # This suggests it is NOT a valid single JSON list, but a file where each line is a JSON object (JSONL).
        pass

def load_data(file_path):
    articles = []
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try reading as standard JSON first
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # If that fails, try line-by-line (NDJSON/JSONL)
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return articles

import re

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def create_text_chunks(articles):
    chunks = []
    print("Processing articles...")
    for art in articles:
        # Extract fields
        title = art.get('title', '')
        lead = clean_html(art.get('lead', ''))
        
        content = clean_html(art.get('content', ''))
        
        # Construct text representation with more content
        text = f"Title: {title}\nLead: {lead}"
        if content:
             text += f"\nContent: {content[:4000]}" # Increased limit to 4000 chars
        
        # Extract timestamp
        publish_raw = art.get('publish', {})
        timestamp = 0
        if isinstance(publish_raw, dict) and '$date' in publish_raw:
             val = publish_raw['$date']
             if isinstance(val, dict) and '$numberLong' in val:
                 timestamp = int(val['$numberLong'])
             elif isinstance(val, str): # sometimes numberLong is a string in raw json load
                 timestamp = int(val)
             elif isinstance(val, int):
                 timestamp = int(val)
        
        chunk = {
            'id': str(art.get('_id', {}).get('$oid', 'unknown')),
            'title': title,
            'url': f"https://tanarblog.hu/cikkek/{art.get('alias', '')}", 
            'text': text,
            'timestamp': timestamp
        }
        chunks.append(chunk)
    return chunks

def ingest():
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found.")
        return

    articles = load_data(JSON_FILE)
    print(f"Found {len(articles)} articles.")

    chunks = create_text_chunks(articles)
    
    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings (this may take a while)...")
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save everything
    store = {
        'chunks': chunks,
        'embeddings': embeddings
    }
    
    with open(VECTOR_STORE_FILE, 'wb') as f:
        pickle.dump(store, f)
    
    print(f"Successfully saved {len(chunks)} embeddings to {VECTOR_STORE_FILE}")

if __name__ == "__main__":
    ingest()
