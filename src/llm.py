import requests
from .config import LLM_MODEL,LLM_TIMEOUT,LLM_URL

def generate_answer(prompt):
    response = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 128
            }
        },
        timeout=LLM_TIMEOUT
    )

    response.raise_for_status()
    try:
        data = response.json()
        answer = data['response']
    except (ValueError, KeyError) as e:
        raise RuntimeError("LLM API 返回的数据格式异常") from e

    # print("生成 Token 数：", data.get('eval_count'))
    # print("生成耗时：", data.get('eval_duration') / 1e9, "秒")

    return answer

if __name__ == "__main__":
    pass