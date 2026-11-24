import textwrap
import re

# 基于固定长度进行文本切分
def text_chunking(text, chunk_size = 150, overlap = 30):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = textwrap.dedent(text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def get_overlap_sentences(sentences, overlap = 20):
    if overlap == 0:
        return []
    overlap_sentences = []
    current_chunk = ""
    for i in sentences[::-1]:
        overlap_sentences.append(i)
        current_chunk = i + current_chunk
        if len(current_chunk) >= overlap:
            break
    return overlap_sentences[::-1]

# 基于句子进行chunk
def chunk_text_by_sentences(text, chunk_size=30, overlap=5):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not text.strip():
        return []
    
    sentences = re.split(r'(?<=[。！？])', text)  # 按中文句号、问号、感叹号分割
    sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白句子
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # 对于超长句子进行二次切分，使用固定长度进行切分
        if len(sentence) > chunk_size:
            # 先保存之前积累的普通文本
            if chunk_size <= overlap:
                raise ValueError("chunk_size must be greater than overlap for long sentences.")
            if current_chunk:
                chunks.append("".join(current_chunk))
            long_chunks = []
            step = chunk_size - overlap
            for i in range(0, len(sentence), step):
                chunk = sentence[i:i + chunk_size]
                if not chunk:
                    break
                long_chunks.append(chunk)
                if i + chunk_size >= len(sentence):
                    break
            # 保存所有切片
            chunks.extend(long_chunks)
            # 清空当前状态
            current_chunk = []
            continue
        # 对于正常长度的句子处理，直接利用句子边界进行处理
        current_text = ''.join(current_chunk)
        if len(current_text) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(current_text)
                overlap_sentences = get_overlap_sentences(current_chunk, overlap)
                current_chunk = overlap_sentences
            current_chunk.append(sentence)
    # 保留最后一个chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks

# pdf每页切片
def chunk_pdf_pages(pages, chunk_size = 150, overlap = 30):
    result = []

    for item in pages:
        page = item['page']
        text = item['text']
        chunks = chunk_text_by_sentences(
            text,
            chunk_size = chunk_size,
            overlap = overlap
        )

        for chunk in chunks:
            result.append({
                'page': page,
                'text':chunk
            })

    return result


if __name__ == "__main__":
    pass