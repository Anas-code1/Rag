import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings


def compute_cosine_similarity(query_vector: np.ndarray, doc_embeddings: np.ndarray) ->np.ndarray:
    query_vector = query_vector.flatten()
    dot_product = np.dot(doc_embeddings, query_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)

    similarity_score = dot_product / (query_norm * doc_norms)
    return similarity_score

if __name__ == "__main__":
    pdf_file = "sample.pdf"

    raw_text = extract_text_from_pdf(pdf_file)
    document_chunks = chunk_text(raw_text, chunk_size=1000, overlap=50)
    chunk_embeddings = generate_embeddings(document_chunks)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "What are the projects in the document?"
    print(f"Query: {query}")
    query_vector = model.encode([query])
    scores = compute_cosine_similarity(query_vector, chunk_embeddings)  

    best_match_index = np.argmax(scores)
    print(f"\n[+] retrieval results:")
    for i, score in enumerate(scores):
        print(f"Chunk {i+1}: Score: {score:.4f}")

    print(f"\n[+] Best matching chunk (Chunk {best_match_index + 1}):")
    print(document_chunks[best_match_index])