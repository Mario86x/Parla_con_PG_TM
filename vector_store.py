import os
import json
import logging
import sys
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from llm import init_llm, init_embed_model, init_local_embed_model  # tuoi initializer
from tqdm import tqdm
import requests

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PERSIST_DIR = "chroma_storage"
DOCS_PATH = "docs"

load_dotenv()  # carica API keys

def load_json_documents(docs_path: str):
    """
    Carica tutti i file JSON da una cartella e restituisce lista di chunk con metadata.
    """
    all_chunks = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".json"):
            file_path = os.path.join(docs_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, obj in enumerate(data):
                            chunk_id = f"{filename}_{i}"
                            all_chunks.append({
                                "id": chunk_id,
                                "content": obj.get("content", ""),
                                "metadata": obj.get("metadata", {}) | {"source_file": filename}
                            })
                except Exception as e:
                    logging.error(f"Errore caricando {file_path}: {e}")
    return all_chunks

import requests

# --- Embedding function con Ollama ---
def ollama_embedding_function(texts: list[str]) -> list[list[float]]:
    vectors = []
    for t in texts:
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "embeddinggemma:latest", "prompt": t},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            vectors.append(data["embedding"])
        except Exception as e:
            logging.error(f"Errore embedding per testo: {t[:50]}... -> {e}")
            vectors.append([0.0])  # fallback minimo per non bloccare
    return vectors

def create_vector_store(docs_path: str = DOCS_PATH):
    """
    Crea o aggiorna un vector store Chroma a partire da documenti JSON già parsati.
    """

    # Carica documenti JSON
    logging.info(f"Caricamento JSON da {docs_path}")
    chunks = load_json_documents(docs_path)
    logging.info(f"Totale chunk caricati: {len(chunks)}")

    if not chunks:
        raise ValueError("Nessun documento JSON trovato.")

    # Crea client Chroma persistente
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Recupera o crea collezione
    collection = chroma_client.get_or_create_collection(
        name="documents",
        # embedding_function=ollama_embedding_function,
        metadata={"description": "Base conoscenza Ardania"}
    )

    # Aggiungi i chunk alla collezione
    logging.info("Inserimento chunk nella collezione Chroma...")
    for chunk in tqdm(chunks, desc="Inserting Chunks"):
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["content"]],
            metadatas=[chunk["metadata"]],
            embeddings=ollama_embedding_function([chunk["content"]])
        )

    logging.info("Inserimento completato.")
    return collection

if __name__ == "__main__":
    try:
        collection = create_vector_store()
        print(f"Vector store creato/aggiornato e salvato in '{PERSIST_DIR}'")
    except Exception as e:
        print(f"Errore nella creazione del vector store: {e}")
