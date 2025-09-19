import hashlib
from tqdm import tqdm
import chromadb
import requests
import logging
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Any

# Parametri ottimizzati
CHUNK_SIZE = 500
OVERLAP = 50
BATCH_SIZE = 50  # Aumentato per ridurre le chiamate API
EMBEDDING_BATCH_SIZE = 20  # Batch per embeddings paralleli
MAX_WORKERS = 5  # Thread per embedding paralleli

KNOWLEDGE_PATH = "lore_md"  # changed from knowledge_md
PERSIST_DIR = "chroma_lore_db"  # changed from chroma_db
COLLECTION_NAME = "ardania_lore"  # changed from ardania_knowledge



def count_lines_in_file(file_path):
    """Conta il numero di righe in un file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)


def init_chroma_collection(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME):
    """
    Inizializza ChromaDB persistente e ritorna la collezione della lore.
    """
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Base conoscenza della lore di Ardania"}
    )
    logging.info(f"Collezione lore '{collection_name}' pronta in '{persist_dir}'")
    return collection


async def ollama_embedding_function_async(texts: List[str], session: aiohttp.ClientSession) -> List[List[float]]:
    """Versione asincrona per embedding paralleli"""
    async def get_embedding(text: str) -> List[float]:
        try:
            async with session.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "embeddinggemma:latest", "prompt": text},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["embedding"]
        except Exception as e:
            logging.error(f"Errore embedding per testo: {text[:50]}... -> {e}")
            raise e
    
    # Esegui tutti gli embedding in parallelo
    tasks = [get_embedding(text) for text in texts]
    return await asyncio.gather(*tasks)


def ollama_embedding_function_threaded(texts: List[str]) -> List[List[float]]:
    """Versione con thread pool per embedding paralleli (sync)"""
    def get_single_embedding(text: str) -> List[float]:
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "embeddinggemma:latest", "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logging.error(f"Errore embedding per testo: {text[:50]}... -> {e}")
            raise e
    
    # Usa ThreadPoolExecutor per parallelizzare
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_single_embedding, text) for text in texts]
        return [future.result() for future in futures]


def preprocess_markdown_file(md_path: str) -> List[Dict[str, Any]]:
    """
    Pre-processa tutto il file markdown in memoria per creare i chunk,
    evitando elaborazioni ridondanti durante l'inserimento.
    """
    chunks = []
    order = 0
    buffer = ""
    headings = []
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    logging.info(f"Pre-processando {total_lines} righe da {md_path}...")
    
    for line in tqdm(lines, desc="Pre-processing"):
        line = line.strip()
        if line.startswith("#"):
            level = len(line.split(" ")[0])
            title = line[level:].strip()
            
            # Aggiorna headings
            if len(headings) < level:
                headings.extend([""] * (level - len(headings)))
            headings[level - 1] = title
            for i in range(level, len(headings)):
                headings[i] = ""
        else:
            if line:
                buffer += (" " + line).strip()
                
                # Crea chunk quando raggiunge la dimensione
                while len(buffer) >= CHUNK_SIZE:
                    order += 1
                    text_block = buffer[:CHUNK_SIZE]
                    hash_id = hashlib.sha256(text_block.encode("utf-8")).hexdigest()
                    heading_str = " > ".join([h for h in headings if h])
                    
                    chunk = {
                        "id": hash_id,
                        "content": text_block,
                        "metadata": {
                            "headings": heading_str,
                            "hash": hash_id,
                            "order": order,
                            "file": os.path.basename(md_path)
                        }
                    }
                    chunks.append(chunk)
                    buffer = buffer[CHUNK_SIZE - OVERLAP:]
    
    # Gestisci ultimo buffer
    if buffer.strip():
        order += 1
        hash_id = hashlib.sha256(buffer.encode("utf-8")).hexdigest()
        heading_str = " > ".join([h for h in headings if h])
        
        chunk = {
            "id": hash_id,
            "content": buffer,
            "metadata": {
                "headings": heading_str,
                "hash": hash_id,
                "order": order,
                "file": os.path.basename(md_path)
            }
        }
        chunks.append(chunk)
    
    logging.info(f"✅ Pre-processing completato: {len(chunks)} chunk generati")
    return chunks


def batch_insert_chunks(chunks: List[Dict[str, Any]], collection, batch_size=BATCH_SIZE):
    """
    Inserisce i chunk in batch ottimizzati con embedding paralleli.
    """
    total_chunks = len(chunks)
    logging.info(f"Inserimento {total_chunks} chunk in batch di {batch_size}...")
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Inserimento batch"):
        batch = chunks[i:i + batch_size]
        
        # Calcola embedding in parallelo
        start_time = time.time()
        texts = [chunk["content"] for chunk in batch]
        embeddings = ollama_embedding_function_threaded(texts)
        embed_time = time.time() - start_time
        
        # Inserisci in ChromaDB
        start_time = time.time()
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in batch],
            embeddings=embeddings
        )
        insert_time = time.time() - start_time
        
        logging.debug(f"Batch {i//batch_size + 1}: embedding={embed_time:.2f}s, insert={insert_time:.2f}s")


async def batch_insert_chunks_async(chunks: List[Dict[str, Any]], collection, batch_size=BATCH_SIZE):
    """
    Versione asincrona per inserimento batch con controllo ID esistenti.
    """
    total_chunks = len(chunks)
    logging.info(f"Inserimento asincrono {total_chunks} chunk in batch di {batch_size}...")
    
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, total_chunks, batch_size), desc="Inserimento async"):
            batch = chunks[i:i + batch_size]
            batch_ids = [chunk["id"] for chunk in batch]
            
            # Controlla quali ID esistono già
            start_time = time.time()
            try:
                existing_docs = collection.get(ids=batch_ids)
                existing_ids = set(existing_docs['ids']) if existing_docs['ids'] else set()
            except Exception:
                existing_ids = set()
            check_time = time.time() - start_time
            
            # Filtra solo i chunk nuovi
            new_chunks = [chunk for chunk in batch if chunk["id"] not in existing_ids]
            existing_count = len(batch) - len(new_chunks)
            
            if existing_count > 0:
                logging.info(f"Saltati {existing_count} chunk già esistenti")
            
            if not new_chunks:
                continue
            
            # Calcola embedding in parallelo (async) solo per chunk nuovi
            start_time = time.time()
            new_texts = [chunk["content"] for chunk in new_chunks]
            embeddings = await ollama_embedding_function_async(new_texts, session)
            embed_time = time.time() - start_time
            
            # Inserisci in ChromaDB (sync, ma veloce)
            start_time = time.time()
            collection.upsert(
                ids=[chunk["id"] for chunk in new_chunks],
                documents=new_texts,
                metadatas=[chunk["metadata"] for chunk in new_chunks],
                embeddings=embeddings
            )
            insert_time = time.time() - start_time
            
            logging.debug(f"Batch async {i//batch_size + 1}: check={check_time:.2f}s, embedding={embed_time:.2f}s, insert={insert_time:.2f}s, nuovi={len(new_chunks)}/{len(batch)}")


def process_file_optimized(md_path: str, collection, use_async=False):
    """
    Processa un singolo file markdown in modo ottimizzato.
    """
    logging.info(f"🚀 Processing file ottimizzato: {os.path.basename(md_path)}")
    
    # Pre-processa tutto il file
    chunks = preprocess_markdown_file(md_path)
    
    if not chunks:
        logging.warning(f"Nessun chunk generato per {md_path}")
        return
    
    # Inserisci in batch
    if use_async:
        asyncio.run(batch_insert_chunks_async(chunks, collection))
    else:
        batch_insert_chunks(chunks, collection)
    
    logging.info(f"✅ File completato: {os.path.basename(md_path)} ({len(chunks)} chunk)")


def main():
    """Main function ottimizzata"""
    start_time = time.time()
    
    # Inizializza la collezione Chroma
    collection = init_chroma_collection()
    
    # Trova tutti i file markdown
    md_files = [f for f in os.listdir(KNOWLEDGE_PATH) if f.endswith(".md")]
    logging.info(f"🔍 Trovati {len(md_files)} file markdown in '{KNOWLEDGE_PATH}'")
    
    # Processa ogni file
    for md_file in md_files:
        md_path = os.path.join(KNOWLEDGE_PATH, md_file)
        file_start = time.time()
        
        # Usa versione asincrona per performance massime
        process_file_optimized(md_path, collection, use_async=True)
        
        file_time = time.time() - file_start
        logging.info(f"⏱️  Tempo file {md_file}: {file_time:.2f} secondi")
    
    total_time = time.time() - start_time
    logging.info(f"🎉 Elaborazione completata in {total_time:.2f} secondi!")
    
    # Statistiche finali
    total_docs = collection.count()
    logging.info(f"📊 Documenti totali nella collezione: {total_docs}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()