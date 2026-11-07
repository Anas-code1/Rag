import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from ingest import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from search import compute_cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Initialize the official Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_rag_answer(query: str, retrieved_chunk: str) -> str:
    """
    Constructs an augmented prompt with the retrieved context 
    and calls Gemini to generate a grounded answer.
    """
    system_instruction = (
        "You are an expert AI assistant. Answer the user's question "
        "strictly using ONLY the provided document context below. "
        "If the answer cannot be found in the context, state clearly "
        "that it is not mentioned in the document."
    )
    
    prompt = f"""
[Document Context]
{retrieved_chunk}

[User Question]
{query}
"""

    print("[*] Sending augmented prompt to Gemini...")
    
    # Call Gemini 2.5 Flash using the modern GenAI SDK
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'system_instruction': system_instruction,
            'temperature': 0.0 # Factual and precise
        }
    )
    
    content = response.text
    return content if content is not None else "Error: Model returned empty content."

if __name__ == "__main__":
    pdf_file = "sample.pdf"
    
    # 1. Ingestion & Embedding Pipeline
    print("[*] Running ingestion & embedding pipeline...")
    raw_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    chunk_embeddings = generate_embeddings(chunks)
    
    # 2. Setup the embedding model for the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Define your user question
    query =  input("Ask a question about the document: ")
    print(f"\n[*] User Query: '{query}'")
    
    # 4. Retrieval Phase (Cosine Similarity)
    query_vector = model.encode(query)
    scores = compute_cosine_similarity(query_vector, chunk_embeddings)
    
    best_match_idx = np.argmax(scores)
    best_chunk = chunks[best_match_idx]
    
    print(f"[+] Retrieved Best Chunk (Index {best_match_idx} | Score: {scores[best_match_idx]:.4f})")
    
    # 5. Generation Phase (Gemini API Call)
    answer = generate_rag_answer(query, best_chunk)
    
    print("\n" + "="*40)
    print("🤖 FINAL RAG GENERATED ANSWER (VIA GEMINI):")
    print("="*40)
    print(answer)
    print("="*40)