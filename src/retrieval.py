import numpy as np

def cosine_similarity(a, b):    #基于numpy的余弦相似度计算
    a = np.array(a)
    b = np.array(b)

    demominator = np.linalg.norm(a) * np.linalg.norm(b)

    if demominator == 0:
        return 0.0

    return float(np.dot(a, b) / demominator)

# 基于余弦相似度的检索
def retrieve_top_k(chunks, embeddings, query_embedding, k=2, threshold=0.5):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")    # 确保k为正整数    

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")    # 1. 检查 chunks 和 embeddings 数量是否一致

    # 检查问题向量和文档向量维度是否一致
    if not embeddings:
        return []
    query_dim = len(query_embedding)
    # 问题向量为空时直接抛出异常，不进行计算
    if query_dim == 0:
        raise ValueError("Query embedding is empty")
    
    for vector in embeddings:
        if len(vector) != query_dim:
            raise ValueError("Query embedding dimension does not match document embeddings")
        
    result = []      # 2. 创建一个列表，保存每个文本块及其相似度

    for index, (text, vector) in enumerate(zip(chunks, embeddings)):        # 3. 遍历 embeddings
        score = cosine_similarity(vector,query_embedding)       #    计算每个向量和 query_embedding 的相似度
        if score >= threshold:        #    如果相似度大于等于阈值，则保存文本块、相似度和原始索引
            result.append({'text': text,     # 4. 保存文本块、相似度和原始索引
                        'score': score,
                        'index':index})
        
    result.sort(key = lambda item: item['score'], reverse= True)     # 5. 按相似度从高到低排序

    return result[:k]        # 6. 返回前 k 个结果

if __name__ == "__main__":
    pass