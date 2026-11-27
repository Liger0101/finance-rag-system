from src.ingestion import load_and_chunk_pdf, ingest_pdf
import pymupdf
import pytest

# 测试加载
def test_load_and_chunk_pdf():
    file_path = 'data/raw/haier_2024_annual_report.pdf'
    result = load_and_chunk_pdf(file_path)

    assert isinstance(result, list)
    assert len(result)>0

    result_index = {
        'page',
        'text'
    }

    for i in result:
        assert isinstance(i,dict)
        assert result_index.issubset(i.keys())
        assert isinstance(i['page'], int)
        assert i['page'] > 0
        assert isinstance(i['text'], str)
        assert i['text'].strip() != ''

# 测试参数传递
def test_load_and_chunk_pdf_parameters(monkeypatch):
    captured = {}

    def mock_load_pdf(file_path):
        return [{'page': 1, 'text': '测试内容。'}]

    def mock_chunk_pdf_pages(pages, chunk_size, overlap):
        captured['chunk_size'] = chunk_size
        captured['overlap'] = overlap
        return [{'page': 1, 'text': '测试内容。'}]

    monkeypatch.setattr('src.ingestion.load_pdf', mock_load_pdf)
    monkeypatch.setattr('src.ingestion.chunk_pdf_pages', mock_chunk_pdf_pages)

    result = load_and_chunk_pdf(
        'test.pdf',
        chunk_size=200,
        overlap=40
    )

    assert captured['chunk_size'] == 200
    assert captured['overlap'] == 40
    assert len(result) == 1

# 异常文件处理
def test_load_and_chunk_padf_file_not_found():
    with pytest.raises(pymupdf.FileNotFoundError):
        load_and_chunk_pdf('data/raw/not_exist.pdf')

# ingest正常测试
def test_ingest_pdf(monkeypatch):
    chunks = [
        {'page': 1, 'text': '测试内容。'}
    ]

    embedded_chunks = [
        {
            'page': 1,
            'text': '测试内容。',
            'embedding': [0.1, 0.2, 0.3]
        }
    ]

    monkeypatch.setattr(
        'src.ingestion.load_and_chunk_pdf',
        lambda *args, **kwargs: chunks
    )

    monkeypatch.setattr(
        'src.ingestion.embed_chunks',
        lambda data: embedded_chunks
    )

    result = ingest_pdf('test.pdf')

    assert result == embedded_chunks
    assert 'embedding' in result[0]

# 测试传参
def test_ingest_pdf_parameters(monkeypatch):
    captured = {}

    def mock_load_and_chunk_pdf(file_path, chunk_size, overlap):
        captured['file_path'] = file_path
        captured['chunk_size'] = chunk_size
        captured['overlap'] = overlap
        return [{'page': 1, 'text': '测试内容。'}]

    def mock_embed_chunks(chunks):
        return chunks

    monkeypatch.setattr(
        'src.ingestion.load_and_chunk_pdf',
        mock_load_and_chunk_pdf
    )

    monkeypatch.setattr(
        'src.ingestion.embed_chunks',
        mock_embed_chunks
    )

    ingest_pdf('test.pdf', chunk_size=200, overlap=40)

    assert captured['file_path'] == 'test.pdf'
    assert captured['chunk_size'] == 200
    assert captured['overlap'] == 40