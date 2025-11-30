import redis
import numpy as np
import hashlib
import requests
import logging
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# --- Configurazione Globale ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DOC_INDEX_NAME = "ardania_lore_idx"
DOC_PREFIX = "doc:"

OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "embeddinggemma:latest"
VECTOR_DIM = 768

KNOWLEDGE_PATH = "lore_md"
CHUNK_SIZE = 1500
OVERLAP = 150
BATCH_SIZE = 5
MAX_WORKERS = 5

logger = logging.getLogger("RedisLoreEmbedder")


def get_redis_client() -> redis.Redis | None:
    """Tenta la connessione a Redis e restituisce il client."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        r.ping()
        logger.info("✅ Connessione a Redis riuscita!")
        return r
    except redis.exceptions.ConnectionError as e:
        logger.error(f"❌ ERRORE: Impossibile connettersi a Redis. Dettagli: {e}")
        return None


def init_redis_index(r: redis.Redis):
    """
    Definisce e crea l'indice dei documenti (Vettore e Testo) in RediSearch.

    CORREZIONE PRINCIPALE:
    - L'import corretto per IndexDefinition/IndexType è da
      redis.commands.search.index_definition (non indexDefinition).
    - Anche i Field si definiscono passando il nome del campo come primo argomento.
    """
    # imports corretti (nota: index_definition -> underscore)
    from redis.commands.search.field import TextField, TagField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType

    # Tentativo di eliminare l'indice esistente (gestione errori più robusta)
    try:
        try:
            r.ft(DOC_INDEX_NAME).dropindex(delete_documents=False)
            logger.info(f"⚠️ Vecchio indice '{DOC_INDEX_NAME}' eliminato.")
        except Exception as e_drop:
            # se non esiste o errore, logga e continua
            logger.debug(f"Nessun indice da eliminare o errore durante dropindex: {e_drop}")
    except Exception as e:
        logger.debug(f"Ignoro errore dropindex: {e}")

    # Definizione dello Schema (usare i nomi dei campi esatti)
    schema = [
        TextField("text", weight=1.0),
        TextField("headings"),
        TagField("file"),
        VectorField("vector", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE"
        })
    ]

    # Creazione dell'indice
    definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
    try:
        r.ft(DOC_INDEX_NAME).create_index(schema, definition=definition)
        logger.info(f"✅ Indice '{DOC_INDEX_NAME}' creato con successo.")
    except Exception as e:
        logger.error(f"❌ ERRORE creazione indice: {e}")
        raise
    return r


def ollama_embedding_function_threaded(texts: List[str]) -> List[List[float]]:
    """Versione con thread pool per embedding paralleli (sync)"""
    def get_single_embedding(text: str) -> List[float]:
        try:
            response = requests.post(
                OLLAMA_EMBEDDING_URL,
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # il payload di Ollama può variare: assumo `embedding` diretto
            return data["embedding"]
        except Exception as e:
            logger.error(f"Errore embedding per testo: {text[:50]}... -> {e}")
            return [0.0] * VECTOR_DIM

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_single_embedding, text) for text in texts]
        return [future.result() for future in futures]


def preprocess_markdown_file(md_path: str) -> List[Dict[str, Any]]:
    """
    Pre-processa il file markdown e crea i chunk.
    """
    chunks = []
    order = 0
    buffer = ""
    headings: List[str] = []

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Pre-processando {len(lines)} righe da {os.path.basename(md_path)}...")

    for line in tqdm(lines, desc="Pre-processing"):
        line = line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("#"):
            # calcola il livello (numero di #)
            level = len(stripped.split(" ")[0])
            title = stripped[level:].strip()

            if len(headings) < level:
                headings.extend([""] * (level - len(headings)))
            headings[level - 1] = title
            # azzera i livelli più profondi
            for i in range(level, len(headings)):
                headings[i] = ""
        else:
            if stripped:
                if buffer:
                    buffer += " " + stripped
                else:
                    buffer = stripped

                while len(buffer) >= CHUNK_SIZE:
                    order += 1
                    text_block = buffer[:CHUNK_SIZE]
                    hash_id = hashlib.sha256(text_block.encode("utf-8")).hexdigest()
                    heading_str = " > ".join([h for h in headings if h])

                    chunks.append({
                        "id": hash_id,
                        "content": text_block,
                        "metadata": {
                            "headings": heading_str,
                            "order": order,
                            "file": os.path.basename(md_path)
                        }
                    })
                    buffer = buffer[CHUNK_SIZE - OVERLAP:]

    if buffer.strip():
        order += 1
        hash_id = hashlib.sha256(buffer.encode("utf-8")).hexdigest()
        heading_str = " > ".join([h for h in headings if h])

        chunks.append({
            "id": hash_id,
            "content": buffer,
            "metadata": {
                "headings": heading_str,
                "order": order,
                "file": os.path.basename(md_path)
            }
        })

    logger.info(f"✅ Pre-processing completato: {len(chunks)} chunk generati")
    return chunks


def insert_chunks_to_redis(r: redis.Redis, chunks: List[Dict[str, Any]], batch_size=BATCH_SIZE):
    """
    Inserisce i chunk e i loro embeddings su Redis Stack.
    """
    total_chunks = len(chunks)
    logger.info(f"Inserimento {total_chunks} chunk in batch di {batch_size} su Redis...")

    for i in tqdm(range(0, total_chunks, batch_size), desc="Inserimento batch Redis"):
        batch = chunks[i:i + batch_size]

        # 1. Calcola embedding in parallelo
        texts = [chunk["content"] for chunk in batch]
        embeddings = ollama_embedding_function_threaded(texts)

        # 2. Inserisci in Redis tramite pipeline
        pipe = r.pipeline()
        for chunk, embedding in zip(batch, embeddings):
            key = f"{DOC_PREFIX}{chunk['id']}"

            data_to_store = {
                # hset richiede bytes o string; usiamo bytes per text e headings
                "text": chunk["content"].encode("utf-8"),
                "headings": chunk["metadata"]["headings"].encode("utf-8"),
                "file": chunk["metadata"]["file"].encode("utf-8"),
                # Il vettore come float32 bytes
                "vector": np.array(embedding, dtype=np.float32).tobytes()
            }
            pipe.hset(key, mapping=data_to_store)

        pipe.execute()
        logger.debug(f"Batch {i//batch_size + 1} inserito.")


def process_file_optimized(md_path: str, r: redis.Redis):
    logger.info(f"🚀 Processing file ottimizzato: {os.path.basename(md_path)}")

    chunks = preprocess_markdown_file(md_path)

    if not chunks:
        logger.warning(f"Nessun chunk generato per {md_path}")
        return

    insert_chunks_to_redis(r, chunks)
    logger.info(f"✅ File completato: {os.path.basename(md_path)} ({len(chunks)} chunk)")


def main():
    start_time = time.time()

    r = get_redis_client()
    if r is None:
        return

    try:
        init_redis_index(r)
    except Exception as e:
        logger.error(f"Impossibile inizializzare indice Redis: {e}")
        return

    if not os.path.isdir(KNOWLEDGE_PATH):
        logger.error(f"Directory '{KNOWLEDGE_PATH}' non trovata.")
        return

    md_files = [f for f in os.listdir(KNOWLEDGE_PATH) if f.endswith(".md")]
    logger.info(f"🔍 Trovati {len(md_files)} file markdown in '{KNOWLEDGE_PATH}'")

    for md_file in md_files:
        md_path = os.path.join(KNOWLEDGE_PATH, md_file)
        file_start = time.time()
        process_file_optimized(md_path, r)
        file_time = time.time() - file_start
        logger.info(f"⏱️  Tempo file {md_file}: {file_time:.2f} secondi")

    total_time = time.time() - start_time
    logger.info(f"🎉 Elaborazione completata in {total_time:.2f} secondi!")

    try:
        info = r.ft(DOC_INDEX_NAME).info()
        total_docs = info.get('num_docs', 'n/a')
        logger.info(f"📊 Documenti totali indicizzati in Redis: {total_docs}")
    except Exception:
        logger.warning("Impossibile recuperare il conteggio finale dei documenti da Redis.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
