import requests
from .config import EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_TIMEOUT

def text_embedding(chunks: list): 
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": chunks
            },
            timeout=EMBEDDING_TIMEOUT
        )

        response.raise_for_status()
        try:
            data = response.json()['embeddings']
        except(KeyError, ValueError) as e:
            raise RuntimeError('Embedding API 返回的数据格式异常') from e
        return data
    
    except requests.exceptions.Timeout as e:
        raise RuntimeError("Embedding 请求超时") from e

    except requests.exceptions.ConnectionError as e:
        raise RuntimeError("无法连接 Ollama，请检查服务是否启动") from e

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Embedding API 请求失败：{response.status_code} - {response.text}"
        ) from e

def embed_chunks(chunks):
    texts = []
    for page in chunks:
        texts.append(page['text'])

    data = text_embedding(texts)

    if len(chunks) != len(data):
        raise ValueError("Embedding 数量与 Chunk 数量不一致")

    new_chunks = []

    for chunk, embed in zip(chunks, data):
        new_chunk = {
            'page': chunk['page'],
            'text': chunk['text'],
            'embedding': embed
        }
        new_chunks.append(new_chunk)
    
    return new_chunks

if __name__ == "__main__":
    pass