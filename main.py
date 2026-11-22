from src.chunking import text_chunking, chunk_text_by_sentences
from src.embeddings import text_embedding
from src.retrieval import retrieve_top_k
from src.prompting import build_prompt
from src.llm import generate_answer

def main():
    text = """
    2025年公司营业收入为100亿元。
    2025年公司净利润为12亿元。
    2024年公司营业收入为85亿元。
    公司总部位于上海。
    """
    query = "2025年公司的员工总人数是多少？"

    chunks = chunk_text_by_sentences(
        text,
        chunk_size=15,
        overlap=5
    )

    embedding = text_embedding(chunks)
    query_embedding = text_embedding([query])[0]

    result = retrieve_top_k(
        chunks,
        embedding,
        query_embedding,
        k=4
    )

    prompt = build_prompt(result, query)
    answer = generate_answer(prompt)

    print('检索结果为:')
    print(result)
    print('\n生成的回答为:')
    print(answer) 

if __name__ == '__main__':
    main()