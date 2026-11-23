from evaluation.load_dataset import load_dataset
import json

# 数据集正常加载
def test_load_dataset(tmp_path):
    data = [
        {
            "question": "公司总部在哪里？",
            "question_type": "事实查询",
            "expected_answer": "山东青岛",
            "evidence": "公司基本情况",
            "source_page": 13
        }
    ]

    file_path = tmp_path/'test_tmp.json'

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    result = load_dataset(file_path)

    assert result == data

# 测试数据是否为列表，并确定受否有10个数据
def test_dataset_structure():
    data = load_dataset('evaluation/evaluation_dataset.json')

    assert isinstance(data,list)
    assert len(data) == 10

# 是否包含必要字段
def test_dataset_required_fields():
    data = load_dataset('evaluation/evaluation_dataset.json')

    required_fields = {
        'question_type',
        'question',
        'expected_answer',
        'evidence',
        'source_page'
    }

    for i in data:
        assert required_fields.issubset(i.keys())

# 无答案查询测试
def test_no_answer_samples():
    data = load_dataset('evaluation/evaluation_dataset.json')

    no_answer_items = [
        item for item in data
        if item['question_type'] == '无答案查询'
    ]

    assert len(no_answer_items) == 2

    for item in no_answer_items:
        assert item['expected_answer'] is None
        assert item['evidence'] is None
        assert item['source_page'] is None