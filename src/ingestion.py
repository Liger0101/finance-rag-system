from .document_load import load_pdf
from .chunking import chunk_pdf_pages
from .embeddings import embed_chunks

def load_and_chunk_pdf(file_path, chunk_size = 150, overlap = 30):
    pages = load_pdf(file_path)

    result = chunk_pdf_pages(
        pages, 
        chunk_size=chunk_size,
        overlap=overlap
    )

    return result

def ingest_pdf(file_path, chunk_size = 150, overlap = 30):
    result = load_and_chunk_pdf(
        file_path,
        chunk_size= chunk_size,
        overlap= overlap
    )

    data = embed_chunks(result)

    return data