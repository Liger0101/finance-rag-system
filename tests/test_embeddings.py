import pytest
import requests
from unittest.mock import patch, Mock
from src.embeddings import text_embedding, embed_chunks

# 1.Mock 的作用就是：模拟外部接口的行为，让我们不必真正调用 Ollama，mock模拟数据进行测试
@patch('src.embeddings.requests.post')  # 替换端口
def test_text_embedding(mock_post):
    mock_response = Mock()

    mock_response.json.return_value = { # 设置模拟返回值
        'embeddings':[[0.1, 0.2, 0.3]]
    }

    mock_post.return_value = mock_response

    result = text_embedding(['测试文本'])

    assert result == [[0.1, 0.2, 0.3]]
    mock_post.assert_called_once()

# 2.超时异常处理测试
@patch('src.embeddings.requests.post')
def test_text_embedding_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout # 调用接口时主动抛出异常

    with pytest.raises(RuntimeError, match = 'Embedding 请求超时'):
        text_embedding(['测试文本'])

# 3.ollama 连接异常测试
@patch('src.embeddings.requests.post')
def test_text_embedding_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError

    with pytest.raises(RuntimeError, match = '无法连接 Ollama'):
        text_embedding(['测试文本'])

# 4.Embedding API 返回 HTTP 错误
@patch("src.embeddings.requests.post")
def test_text_embedding_http_error(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError, match="Embedding API 请求失败"):
        text_embedding(["测试文本"])

# 5.测试返回的json文件中是否有embedding的词条
@patch("src.embeddings.requests.post")
def test_text_embedding_missing_embeddings(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        "error": "invalid model"
    }
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError, match = 'Embedding API 返回的数据格式异常'):
        text_embedding(["测试文本"])

# embed_chunk测试
@patch('src.embeddings.text_embedding')
def test_embed_chunks(mock_embedding):
    mock_embedding.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    chunks = [
        {'page': 1, 'text': '测试文本1'},
        {'page': 2, 'text': '测试文本2'}
    ]

    result = embed_chunks(chunks)

    assert len(result) == 2
    assert result[0]['page'] == 1
    assert result[0]['text'] == '测试文本1'
    assert result[0]['embedding'] == [0.1, 0.2, 0.3]

    assert result[1]['page'] == 2
    assert result[1]['embedding'] == [0.4, 0.5, 0.6]

    assert 'embedding' not in chunks[0]
    mock_embedding.assert_called_once_with(['测试文本1', '测试文本2'])

# 测试embedding返回向量和chunk不一致时报错
@patch('src.embeddings.text_embedding')
def test_embed_chunks_embedding_count_mismatch(mock_embedding):
    mock_embedding.return_value = [
        [0.1, 0.2, 0.3]
    ]

    chunks = [
        {'page': 1, 'text': '测试文本1'},
        {'page': 2, 'text': '测试文本2'}
    ]

    with pytest.raises(ValueError, match='Embedding 数量与 Chunk 数量不一致'):
        embed_chunks(chunks)