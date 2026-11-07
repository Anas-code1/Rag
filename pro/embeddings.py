import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import extract_text_from_pdf, chunk_text

def generate_embeddings(chunks: list[str]) -> np.ndarray:
    print("Generating embeddings for document chunks...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    pdf_file = "sample.pdf"

    try:
        raw_text = extract_text_from_pdf(pdf_file)
        print(f"length of extracted text: {len(raw_text)}")

        document_chunks = chunk_text(raw_text, chunk_size=1000, overlap=50)
        print(f"number of document chunks: {len(document_chunks)}")

        embeddings = generate_embeddings(document_chunks)
        print(f"shape of embeddings: {embeddings.shape}")
        print("\n--- Sample Vector for Chunk 1 (First 5 dimensions) ---")
        print(embeddings[0][:5])

    except FileNotFoundError as e:
        print(e)