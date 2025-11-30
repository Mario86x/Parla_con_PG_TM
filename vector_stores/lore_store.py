"""
lore_store.py

Classe LoreStore per interagire con Redis Stack / RediSearch:
- indicizzazione di chunk (text, headings, file, vector)
- ricerca vettoriale nativa (KNN)
- ricerca per keyword (full-text)
- fusione ibrida RRF (Reciprocal Rank Fusion) che combina risultati nativi Redis

Requisiti:
- redis-py recente (4.x) + Redis Stack / RediSearch attivo nel server Redis
- numpy, requests
- endpoint Ollama (o adatta `get_embedding` se vuoi usare altro)
"""

from __future__ import annotations
import logging
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple

import redis
import numpy as np
import requests
from redis.commands.search.query import Query

logger = logging.getLogger("LoreStore")
logger.addHandler(logging.NullHandler())


class LoreStore:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        doc_prefix: str = "doc:",
        index_name: str = "ardania_lore_idx",
        vector_field: str = "vector",
        vector_dim: int = 768,
        ollama_url: str = "http://localhost:11434/api/embeddings",
        ollama_model: str = "embeddinggemma:latest",
        decode_responses: bool = False,
    ):
        """
        Inizializza il LoreStore.

        Parametri:
            redis_host, redis_port: connessione Redis
            doc_prefix: prefisso chiavi hash
            index_name: nome indice RediSearch
            vector_field: nome campo vettoriale nell'indice
            vector_dim: dimensione attesa degli embedding
            ollama_url, ollama_model: per get_embedding() (puoi sostituire o sovrascrivere il metodo)
            decode_responses: setta a False per lavorare con bytes (safe_decode gestirà entrambi)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.doc_prefix = doc_prefix
        self.index_name = index_name
        self.vector_field = vector_field
        self.vector_dim = vector_dim
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        self.client: Optional[redis.Redis] = None
        self.decode_responses = decode_responses

    # -------------------------
    # Connessione e Indexing
    # -------------------------
    def connect(self) -> redis.Redis:
        """Crea e ritorna il client Redis (memorizzato in self.client)."""
        if self.client:
            return self.client
        self.client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=self.decode_responses)
        # ping per verificare
        try:
            self.client.ping()
        except Exception as e:
            logger.error("Impossibile connettersi a Redis: %s", e)
            raise
        logger.info("Connessione a Redis OK")
        return self.client

    def init_index(self, recreate: bool = False):
        """
        Crea l'indice RediSearch per i documenti (testo + vector).
        Se recreate=True elimina l'indice esistente (se presente) e lo ricrea.
        """
        r = self.connect()
        # import locale (underscore nel modulo)
        from redis.commands.search.field import TextField, TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        if recreate:
            try:
                r.ft(self.index_name).dropindex(delete_documents=False)
                logger.info("Vecchio indice eliminato (recreate=True).")
            except Exception as e:
                logger.debug("dropindex: %s", e)

        # definizione schema
        schema = [
            TextField("text", weight=1.0),
            TextField("headings"),
            TagField("file"),
            VectorField(self.vector_field, "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": self.vector_dim,
                "DISTANCE_METRIC": "COSINE"
            })
        ]

        definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)
        try:
            r.ft(self.index_name).create_index(schema, definition=definition)
            logger.info("Indice creato: %s", self.index_name)
        except Exception as e:
            # se già esiste, logghiamo e proseguiamo
            logger.warning("Creazione indice fallita (potrebbe già esistere): %s", e)

    def _doc_key(self, doc_id: str) -> str:
        return f"{self.doc_prefix}{doc_id}"

    @staticmethod
    def safe_decode(x: Any) -> Any:
        """Decodifica bytes->str se necessario; se non bytes ritorna l'oggetto così com'è."""
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8")
            except Exception:
                return x
        return x

    def index_chunk(self, chunk: Dict[str, Any]):
        """
        Indicizza un singolo chunk (assume chunk contiene keys: id, content, metadata[file, headings])
        Il campo vector deve essere passato come array/list di float oppure come raw bytes (float32).
        """
        r = self.connect()
        key = self._doc_key(chunk["id"])

        # embedding: ci aspettiamo o embedding già in chunk["embedding"] oppure non presente
        embedding = chunk.get("embedding")
        if embedding is None:
            raise ValueError("Per index_chunk è richiesto 'embedding' nel chunk (lista di float o bytes).")

        # normalizza l'embedding in float32 bytes
        if isinstance(embedding, (list, tuple, np.ndarray)):
            vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
        elif isinstance(embedding, (bytes, bytearray)):
            vec_bytes = bytes(embedding)
        else:
            raise TypeError("Embedding deve essere list/np.ndarray o bytes")

        mapping = {
            "text": chunk["content"].encode("utf-8") if isinstance(chunk["content"], str) else chunk["content"],
            "headings": chunk["metadata"].get("headings", "").encode("utf-8"),
            "file": chunk["metadata"].get("file", "").encode("utf-8"),
            self.vector_field: vec_bytes
        }

        r.hset(key, mapping=mapping)
        logger.debug("Indicizzato chunk key=%s", key)

    def index_chunks_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 16):
        """Indicizza i chunk in batch usando pipeline (attende che ogni chunk abbia 'embedding')."""
        r = self.connect()
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            pipe = r.pipeline()
            for chunk in batch:
                key = self._doc_key(chunk["id"])
                emb = chunk.get("embedding")
                if emb is None:
                    raise ValueError("Ogni chunk deve avere 'embedding' prima dell'indicizzazione.")
                if isinstance(emb, (list, tuple, np.ndarray)):
                    vec = np.array(emb, dtype=np.float32).tobytes()
                else:
                    vec = bytes(emb)
                mapping = {
                    "text": chunk["content"].encode("utf-8") if isinstance(chunk["content"], str) else chunk["content"],
                    "headings": chunk["metadata"].get("headings", "").encode("utf-8"),
                    "file": chunk["metadata"].get("file", "").encode("utf-8"),
                    self.vector_field: vec
                }
                pipe.hset(key, mapping=mapping)
            pipe.execute()
            logger.info("Batch indexed: %d/%d", min(i + batch_size, total), total)

    # -------------------------
    # Embedding (Ollama by default)
    # -------------------------
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Richiede l'embedding al servizio configurato (Ollama di default).
        Ritorna numpy array float32.
        Sovrascrivi questo metodo se vuoi usare un altro provider.
        """
        try:
            resp = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": text},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            emb = np.array(data["embedding"], dtype=np.float32)
            if emb.ndim != 1 or emb.shape[0] != self.vector_dim:
                logger.warning("Embedding shape mismatch: atteso %d ma ottenuto %s", self.vector_dim, emb.shape)
            return emb
        except Exception as e:
            logger.error("Errore get_embedding: %s", e)
            raise

    # -------------------------
    # Metodi di ricerca
    # -------------------------
    def search_vector(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Ricerca vettoriale nativa: calcola embedding della query, fa KNN k e ritorna lista di risultati.
        Ogni risultato è un dict con: id, score (vector_score), text, headings, file, raw
        """
        r = self.connect()
        emb = self.get_embedding(query_text)
        vec_bytes = emb.astype(np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {k} @{self.vector_field} $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("text", "headings", "file", "vector_score")
            .dialect(2)
        )

        res = r.ft(self.index_name).search(q, query_params={"vec": vec_bytes})
        results = []
        for rank, doc in enumerate(res.docs, start=1):
            results.append({
                "id": getattr(doc, "id", None) or getattr(doc, "doc_id", None),
                "rank": rank,
                "score": float(getattr(doc, "vector_score", 0)),
                "text": self.safe_decode(getattr(doc, "text", "")),
                "headings": self.safe_decode(getattr(doc, "headings", "")),
                "file": self.safe_decode(getattr(doc, "file", "")),
                "raw": doc
            })
        return results

    def search_keyword(self, q_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Ricerca per keyword/full-text nativa. Usa la query testuale semplice.
        Ritorna i primi k risultati ordinati per rilevanza di RediSearch.
        """
        r = self.connect()
        # Query semplice: lascia che RediSearch interpreti il testo (puoi personalizzare con operatore @text)
        q = (
            Query(q_text)
            .return_fields("text", "headings", "file")
            .dialect(2)
            .paging(0, k)
        )
        res = r.ft(self.index_name).search(q)
        results = []
        for rank, doc in enumerate(res.docs, start=1):
            results.append({
                "id": getattr(doc, "id", None) or getattr(doc, "doc_id", None),
                "rank": rank,
                # RediSearch fornisce 'score' su risultati testuali solo quando decode_responses=True?
                # Non sempre accessibile, ma il rank è quello che ci serve per RRF.
                "score": getattr(doc, "score", None),
                "text": self.safe_decode(getattr(doc, "text", "")),
                "headings": self.safe_decode(getattr(doc, "headings", "")),
                "file": self.safe_decode(getattr(doc, "file", "")),
                "raw": doc
            })
        return results

    # -------------------------
    # RRF (Reciprocal Rank Fusion) ibrido nativo
    # -------------------------
    def rrf_fuse(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Applica RRF sui due ranking pre-calcolati (semantic_results e keyword_results).
        Formula RRF standard: score = sum(1 / (k + rank_i))
        Dove rank_i parte da 1 per il primo documento in ciascuna lista.
        Ritorna lista ordinata decrescente per score RRF con i campi combinati.
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        def apply_list(res_list: List[Dict[str, Any]]):
            for entry in res_list:
                doc_id = entry["id"]
                rank = entry.get("rank", None)
                if rank is None:
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                if doc_id not in items:
                    items[doc_id] = entry

        apply_list(semantic_results)
        apply_list(keyword_results)

        # ordina per score decrescente
        fused = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            entry = items.get(doc_id, {})
            fused.append({
                "id": doc_id,
                "rrf_score": score,
                "source_text": entry.get("text"),
                "headings": entry.get("headings"),
                "file": entry.get("file"),
                # aggiungo anche i sottoscores se disponibili
                "semantic_rank": next((e["rank"] for e in semantic_results if e["id"] == doc_id), None),
                "keyword_rank": next((e["rank"] for e in keyword_results if e["id"] == doc_id), None)
            })
        return fused

    def hybrid_search(
        self,
        query_text: str,
        top_k_semantic: int = 5,
        top_k_keyword: int = 10,
        final_k: int = 5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Esegue:
          1) ricerca vettoriale nativa (top_k_semantic)
          2) ricerca keyword nativa (top_k_keyword)
          3) combina con RRF e ritorna i top final_k risultati con informazioni utili
        """
        sem = self.search_vector(query_text, k=top_k_semantic)
        kw = self.search_keyword(query_text, k=top_k_keyword)
        fused = self.rrf_fuse(sem, kw, rrf_k=rrf_k)
        return fused[:final_k]

    # -------------------------
    # Utility
    # -------------------------
    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Recupera l'hash dal Redis (hgetall) e decodifica i campi."""
        r = self.connect()
        key = self._doc_key(doc_id)
        data = r.hgetall(key)
        if not data:
            return None
        return {
            k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }

    def delete_doc(self, doc_id: str):
        """Elimina l'hash corrispondente (non elimina documento dall'indice automaticamente)."""
        r = self.connect()
        key = self._doc_key(doc_id)
        r.delete(key)

    def info(self) -> Dict[str, Any]:
        """Ritorna alcune info dell'indice se disponibili."""
        r = self.connect()
        try:
            info = r.ft(self.index_name).info()
            return info
        except Exception as e:
            logger.error("Errore recupero info indice: %s", e)
            return {}

# -------------------------
# Esempio di utilizzo
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = LoreStore()

    # connessione e creazione indice (se necessario)
    store.connect()
    # store.init_index(recreate=True)  # decommenta se vuoi ricreare l'indice

    # Esempio di ricerca ibrida
    q = input("Inserisci query di prova: ")
    results = store.hybrid_search(q, top_k_semantic=5, top_k_keyword=10, final_k=5)
    print("\nRisultati RRF fused:")
    for r in results:
        print(f"- id={r['id']}, rrf_score={r['rrf_score']:.6f}, sem_rank={r.get('semantic_rank')}, kw_rank={r.get('keyword_rank')}")
        print(f"  file: {r.get('file')}")
        print(f"  headings: {r.get('headings')}")
        print(f"  text: {r.get('source_text') if r.get('source_text') else ''}")
        print()
    # results = store.search_vector(q, k=5)  # solo ricerca vettoriale
    # print("\nRisultati ricerca vettoriale:")
    # for r in results:
    #     print(f"- id={r['id']}, score={r['score']:.6f}")
    #     print(f"  file: {r.get('file')}")
    #     print(f"  headings: {r.get('headings')}")
    #     print(f"  text: {r.get('text') if r.get('text') else ''}")
    #     print()
