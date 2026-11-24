from src.document_load import load_pdf
import pymupdf
import pytest

def test_load_pdf():
    text = load_pdf('data/raw/haier_2024_annual_report.pdf')

    assert isinstance(text,list)
    assert len(text)>0
    assert text[0]['page'] == 1 # 确认pdf被成功读取且从1开始

    assert isinstance(text[0]['text'], str)
    assert len(text[0]['text'].strip()) > 0

# 页码测试
def test_pdf_page_numbers():
    text = load_pdf('data/raw/haier_2024_annual_report.pdf')

    pages = [item['page'] for item in text]

    assert pages == list(range(1, len(text) + 1))

# 空白，图片页测试
def test_load_pdf_with_blank_page(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72),'Hello, RAG')
    doc.new_page()

    file_path = tmp_path/'test.pdf'

    doc.save(file_path)
    doc.close()

    text = load_pdf(file_path)

    assert text[1]['text'].strip() == ''
    assert len(text) == 2

# 异常路径处理
def test_load_pdf_file_not_find():
    with pytest.raises(pymupdf.FileNotFoundError):
        load_pdf('data/raw/not_exist.pdf')