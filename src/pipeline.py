from .chunking import chunk_text_by_sentences
from .embeddings import text_embedding
from .retrieval import retrieve_top_k
from .llm import generate_answer
from .prompting import build_prompt

def rag_pipeline(text, query):
    chunks = chunk_text_by_sentences(text)

    embeddings = text_embedding(chunks)
    query_embedding = text_embedding([query])[0]

    result = retrieve_top_k(chunks, embeddings, query_embedding)

    prompt = build_prompt(result, query)

    answer = generate_answer(prompt)

    return answer