import hashlib

def chunk_id_from_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


import numpy as np

def normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)