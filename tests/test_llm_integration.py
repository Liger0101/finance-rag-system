from src.llm import generate_answer

# llm测试
def test_generate_answer_real_ollama():
    prompt = """
    请回答：2+2等于多少？
    请只输出答案，不要解释。
    """

    answer = generate_answer(prompt)

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    assert "4" in answer