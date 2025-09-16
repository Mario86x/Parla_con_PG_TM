import hashlib

def chunk_id_from_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
