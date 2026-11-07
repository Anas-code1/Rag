import os 
from pypdf import PdfReader



def extract_text_from_pdf(pdf_path:str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"could not find pdf at :{pdf_path}")
    
    reader = PdfReader(pdf_path)
    full_text = ""

    print(f"extracting text from pdf")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text: 
            full_text += text
        
    return full_text


def chunk_text(text:str, chunk_size:int=1000, overlap:int = 50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)


    return [c for c in chunks if c]
     


if __name__ == "__main__":
    pdf_file = "sample.pdf"

    try:
        raw_text = extract_text_from_pdf(pdf_file)
        print(f"length of extracted text: {len(raw_text)}")

        document_chunks = chunk_text(raw_text, chunk_size = 1000, overlap = 50)
        print(f"number of document chunks: {len(document_chunks)}")

        

    except FileNotFoundError as e:
        print(e)