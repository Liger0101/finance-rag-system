from src.chunking import chunk_text_by_sentences,text_chunking,chunk_pdf_pages
import pytest   # pytest的使用需要将函数名设置为test_*的格式

# 1.固定长度切分测试
def test_text_chunking():
    text = "A" * 250

    chunks = text_chunking(
        text,
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) == 3 # assert则检查结果是否和预期一致，本句用于判断生成的chunk数
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 90

# 2.异常输入测试，固定长度切分
def test_text_chunking_invalid_chunk_size():
    with pytest.raises(ValueError): # pytest.raises(): 用于检查异常是否按照预期发生
        text_chunking("Hello World", chunk_size=0)  # 测试当chunk大小为0时的异常抛出


def test_text_chunking_invalid_overlap():
    with pytest.raises(ValueError):
        text_chunking("Hello World", chunk_size=100, overlap=100)   # 测试overlap超过chunk大小时的异常抛出

def test_chunk_text_by_sentences():
    text = "公司成立于2000年。公司总部位于上海。公司主营软件开发。"

    chunks = chunk_text_by_sentences(
        text,
        chunk_size=30,
        overlap=5
    )

    assert len(chunks) > 0  # 判断切分的生成
    assert all(isinstance(chunk, str) for chunk in chunks)  # 判断每个chunk都是字符串
    assert "".join(chunks).strip() != ""    # 判断不是空内容

# 3.测试单个句子长度超过 chunk_size 时，函数能否正确处理
def test_chunk_text_by_sentences_long_sentence():
    text = "A" * 80

    chunks = chunk_text_by_sentences(
        text,
        chunk_size=30,
        overlap=5
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "".join(chunks).startswith("A" * 30)

# 页面切分测试
def test_chunk_pdf_pages():
    pages = [
        {'page': 1, 'text': '第一页内容。'},
        {'page': 2, 'text': '第二页内容。'}
    ]

    chunks = chunk_pdf_pages(
        pages,
        chunk_size=150,
        overlap=30
    )

    assert len(chunks) == 2
    assert chunks[0]['page'] == 1
    assert chunks[1]['page'] == 2
    assert all('page' in chunk and 'text' in chunk for chunk in chunks)
    assert all(chunk['text'].strip() != '' for chunk in chunks)

# 单页多chunk测试
def test_chunk_pdf_pages_multiple_chunks():
    pages = [
        {'page': 5, 'text': '公司成立于2000年。' * 20}
    ]

    chunks = chunk_pdf_pages(
        pages,
        chunk_size=30,
        overlap=5
    )

    assert len(chunks) > 1
    assert all(chunk['page'] == 5 for chunk in chunks)
    assert all(isinstance(chunk['text'], str) for chunk in chunks)

# 空白页测试
def test_chunk_pdf_pages_empty_page():
    pages = [
        {'page': 1, 'text': '公司成立于2000年。'},
        {'page': 2, 'text': ''},
        {'page': 3, 'text': '公司总部位于上海。'}
    ]

    chunks = chunk_pdf_pages(pages)

    assert all(chunk['text'].strip() != '' for chunk in chunks)
    assert all(chunk['page'] != 2 for chunk in chunks)
    assert {chunk['page'] for chunk in chunks} == {1, 3}