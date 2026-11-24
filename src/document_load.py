import pymupdf

def load_pdf(file_path):
    text = []

    with pymupdf.open(file_path) as doc:
        for page in range(len(doc)):
            text.append({
                'page': page+1,
                'text': doc[page].get_text()
            })

    return text