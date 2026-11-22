from src.embeddings import text_embedding
import numpy as np

# embedding模型测试
def test_text_embedding_real_ollama():
    texts = [
        "公司2025年营业收入为100亿元。",
        "公司2025年净利润为12亿元。"
    ]

    embeddings = text_embedding(texts)

    assert len(embeddings) == len(texts)    # 测试每条输入文本都有对应向量

    for vector in embeddings:
        assert len(vector) == 1024 # 测试符合当前 Embedding 模型的输出维度
        assert np.all(np.isfinite(vector))  # 测试排除 NaN 和 Inf
        assert np.linalg.norm(vector) > 0   # 测试排除全零向量