from src.llm import generate_answer
from unittest.mock import patch
from src.config import LLM_MODEL, LLM_URL, LLM_TIMEOUT
import pytest
import requests

# 正常测试
@patch("src.llm.requests.post")
def test_generate_answer(mock_post):
    mock_response = mock_post.return_value
    mock_response.json.return_value = {
        "response": "2025年公司净利润为12亿元。"
    }

    answer = generate_answer("公司的净利润是多少？")

    assert answer == "2025年公司净利润为12亿元。"
    mock_post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()

# 验证是否按照预期向ollama发送请求
@patch("src.llm.requests.post")
def test_generate_answer_request_params(mock_post):
    mock_post.return_value.json.return_value = {
        "response": "测试回答"
    }

    generate_answer("测试问题")

    mock_post.assert_called_once_with(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "prompt": "测试问题",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 128
            }
        },
        timeout=LLM_TIMEOUT
    )

# 请求超时测试
@patch("src.llm.requests.post")
def test_generate_answer_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout

    with pytest.raises(requests.exceptions.Timeout):
        generate_answer("测试问题")

# 网络连接异常
@patch("src.llm.requests.post")
def test_generate_answer_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError

    with pytest.raises(requests.exceptions.ConnectionError):
        generate_answer("测试问题")

# http状态码异常
@patch("src.llm.requests.post")
def test_generate_answer_http_error(mock_post):
    mock_response = mock_post.return_value
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError

    with pytest.raises(requests.exceptions.HTTPError):
        generate_answer("测试问题")

# api返回数据格式异常，即判断返回数据中有没有response这个key
@patch("src.llm.requests.post")
def test_generate_answer_invalid_response(mock_post):
    mock_post.return_value.json.return_value = {
        "model": "qwen3.5:9b",
        "done": True
    }

    with pytest.raises(RuntimeError, match="LLM API 返回的数据格式异常"):
        generate_answer("测试问题")

# 验证json解析失败
@patch("src.llm.requests.post")
def test_generate_answer_invalid_json(mock_post):
    mock_post.return_value.json.side_effect = ValueError("Invalid JSON")

    with pytest.raises(RuntimeError, match="LLM API 返回的数据格式异常"):
        generate_answer("测试问题")