import pytest
from src.retrieval import retrieve_top_k, cosine_similarity

# 1.正常检索测试
def test_retrieve_top_k():
    chunks = ["文本A", "文本B", "文本C"]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6]
    ]

    query_embedding = [1.0, 0.0]

    result = retrieve_top_k(
        chunks,
        embeddings,
        query_embedding,
        k=2,
        threshold=0.0
    )

    assert len(result) == 2 # 测试是否返回相似度前二的chunks
    assert result[0]["index"] == 0      # 判断原索引
    assert result[1]["index"] == 2

# 2.测试threshold的阈值过滤
def test_retrieve_top_k_threshold():
    chunks = ["文本A", "文本B", "文本C"]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6]
    ]

    query_embedding = [1.0, 0.0]

    result = retrieve_top_k(
        chunks,
        embeddings,
        query_embedding,
        k=2,
        threshold=0.9
    )

    assert len(result) == 1
    assert result[0]['index'] == 0

# 3.测试检索结果不足topk下返回正常的结果
def test_retrieve_top_k_less_than_k():
    chunks = ["文本A", "文本B", "文本C"]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6]
    ]

    query_embedding = [1.0, 0.0]

    result = retrieve_top_k(
        chunks,
        embeddings,
        query_embedding,
        k=3,
        threshold=0.9
    )

    assert len(result) < 3
    assert len(result) == 1
    assert result[0]['index'] == 0

# 4.空检验
def test_retrieve_top_k_empty_result():
    chunks = ["文本A", "文本B"]

    embeddings = [
        [0.0, 1.0],
        [0.0, 1.0]
    ]

    query_embedding = [1.0, 0.0]

    result = retrieve_top_k(
        chunks,
        embeddings,
        query_embedding,
        k=2,
        threshold=0.5
    )

    assert result == []

# 5.k的合法性检查测试
@pytest.mark.parametrize('k',[0, -1, -5, True, False])  #pytest.mark.parametrize 可以让同一个测试函数使用多组输入执行，不需要为每个错误值单独写一个测试。
def test_retrieve_top_k_invalid_k(k):
    chunks = ['文本1']
    embeddings = [[1.0, 0.0]]
    query_embedding = [[1.0, 0.0]]

    with pytest.raises(ValueError):
        retrieve_top_k(chunks, embeddings, query_embedding, k=k, threshold=0)

# 6.向量维度不一致测试
@pytest.mark.parametrize(
    "embeddings, query_embedding",
    [
        ([[1.0, 0.0, 0.5]], [1.0, 0.0]),
        ([[1.0, 0.0]], [1.0, 0.0, 0.5]),
        ([[1.0, 0.0], [1.0, 0.0, 0.5]], [1.0, 0.0])
    ]
)
def test_retrieve_top_k_dimension_mismatch(embeddings, query_embedding):
    chunks = ["文本A"] * len(embeddings)

    with pytest.raises(ValueError, match="dimension"):
        retrieve_top_k(
            chunks,
            embeddings,
            query_embedding,
            k=1,
            threshold=0.0
        )

# 7.问题维度为0时的测试
def test_retrieve_top_k_empty_query_embedding():
    chunks = ["文本A"]
    embeddings = [[1.0, 0.0]]
    query_embedding = []

    with pytest.raises(ValueError, match="Query embedding is empty"):
        retrieve_top_k(
            chunks,
            embeddings,
            query_embedding,
            k=1,
            threshold=0.0
        )

# 8.文本数目和向量数目不一致的测试
def test_retrieve_top_k_count_mismatch():
    chunks = ["文本A", "文本B", "文本C"]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0]
    ]

    query_embedding = [1.0, 0.0]

    with pytest.raises(ValueError):
        retrieve_top_k(
            chunks,
            embeddings,
            query_embedding,
            k=2,
            threshold=0.0
        )

# 9.文档向量为空的测试
def test_retrieve_top_k_empty_embeddings():
    chunks = ['文本1', '文本2']
    embeddings = []
    query_embedding = [1.0,0.0]

    with pytest.raises(ValueError):
        retrieve_top_k(chunks, embeddings, query_embedding,k = 2, threshold=0)

# 10.零向量测试
@pytest.mark.parametrize(
    "a, b",
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0])
    ]
)
def test_cosine_similarity_zero_vector(a, b):
    result = cosine_similarity(a, b)

    assert result == 0.0

# 11.余弦相似度正常计算测试
@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 1.0], [1.0, 0.0], 2 ** 0.5 / 2)
    ]
)
def test_cosine_similarity_normal(a, b, expected):
    result = cosine_similarity(a, b)

    assert result == pytest.approx(expected)    # pytest.approx()：用于浮点数近似比较，避免浮点计算精度造成测试误判