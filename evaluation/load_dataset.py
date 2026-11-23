import json
from pathlib import Path

def load_dataset(file_path):
    with open(file_path,'r', encoding = 'utf-8') as f:
        data = json.load(f)

    return data