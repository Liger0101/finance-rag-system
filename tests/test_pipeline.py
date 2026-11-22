from src.pipeline import rag_pipeline
from unittest.mock import patch # 添加mock模拟数据,测试函数名称添加integration和mock可用于区分mock测试和真实集成测试，利用-k integration/mock
# # 只运行 Mock 测试
# python -m pytest tests/test_pipeline.py -k mock -v

# # 只运行真实集成测试
# python -m pytest tests/test_pipeline.py -k integration -v

# 1.测试RAG 系统在正常情况下能否完整运行，并正确回答问题。属于真实集成测试
def test_rag_pipeline_integration():
    text = """
    2025年公司营业收入为100亿元。
    2025年公司净利润为12亿元。
    2024年公司营业收入为85亿元。
    """

    query = "2025年公司净利润是多少？"

    answer = rag_pipeline(text, query)

    assert answer is not None
    assert len(answer) > 0
    assert "12" in answer

# 2.测试空检索场景
def test_rag_pipeline_no_relevant_info_integration():
    text = """
    公司成立于2000年。
    公司总部位于上海。
    公司主要从事软件开发业务。
    """

    query = "2025年公司的净利润是多少？"

    answer = rag_pipeline(text, query)

    assert answer is not None
    assert len(answer) > 0
    assert "无法" in answer or "不足" in answer

# 模拟数据测试
@patch("src.pipeline.generate_answer")
@patch("src.pipeline.text_embedding")
@patch("src.pipeline.chunk_text_by_sentences")
def test_rag_pipeline_mock(
    mock_chunking,
    mock_embedding,
    mock_generate
    ):
    mock_chunking.return_value = ["文本A", "文本B"]

    mock_embedding.side_effect = [  # side_effect 就是按照调用顺序，分别返回预设结果。
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0]]
    ]

    mock_generate.return_value = "2025年公司净利润为12亿元。"

    answer = rag_pipeline("原始文档", "公司净利润是多少？")

    assert answer == "2025年公司净利润为12亿元。"

    mock_chunking.assert_called_once_with("原始文档")   #验证文本切分只调用一次，且参数正确

    assert mock_embedding.call_count == 2   # Embedding 分别处理文档和问题，检查调用次数

    mock_generate.assert_called_once()  # LLM 只生成一次回答

    assert mock_embedding.call_args_list[0].args == (["文本A", "文本B"],)   # call_args_list 保存 Mock 对象每一次调用的参数。args是为了获取位置参数

    assert mock_embedding.call_args_list[1].args == (["公司净利润是多少？"],)

    # 获取传递给 LLM 的 Prompt
    prompt = mock_generate.call_args.args[0]    # call_args用于获取最后一次调用

    # 验证 Prompt 包含用户问题和检索参考文本
    assert "公司净利润是多少？" in prompt
    assert "文本A" in prompt or "文本B" in prompt