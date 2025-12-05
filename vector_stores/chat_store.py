from __future__ import annotations
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import redis
import numpy as np
import requests
# Importazioni da RediSearch (come in LoreStore.py)
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

logger = logging.getLogger("ChatStore")
logger.addHandler(logging.NullHandler())

# Configurazione specifica per la chat memory
CHAT_PREFIX = "chat:"
CHAT_INDEX_NAME = "ardania_chat_idx"
DEFAULT_VECTOR_DIM = 768 # Assumiamo la dimensione standard per l'embedding (adattare se necessario)
K_RRF = 60

# --- Funzione di Fusione ---

def apply_rrf(list_a: List[Dict], list_b: List[Dict], k_rrf: int = K_RRF) -> List[Dict]:
    """Applica Reciprocal Rank Fusion (RRF) su due liste di risultati."""
    
    rrf_scores = defaultdict(float)
    doc_map = {} # Mappa ID documento -> Dati completi

    def process_list(results_list):
        for res in results_list:
            doc_id = res['id']
            # Calcolo del punteggio RRF: 1 / (K_RRF + Rank)
            rrf_scores[doc_id] += 1 / (k_rrf + res['rank'])
            # Conserva i dati del documento e il rank originale
            if doc_id not in doc_map:
                 # Conserva l'informazione del rank per poterla stampare nel report
                doc_map[doc_id] = {'data': res, 'sem_rank': None, 'kw_rank': None}
            
            # Segna quale rank ha contribuito
            if res.get('method') == 'semantic':
                doc_map[doc_id]['sem_rank'] = res['rank']
            elif res.get('method') == 'keyword':
                doc_map[doc_id]['kw_rank'] = res['rank']

    process_list(list_a)
    process_list(list_b)

    # Ordina i risultati finali per punteggio RRF decrescente
    final_ranked_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True)
    
    final_results = []
    for doc_id in final_ranked_ids:
        data = doc_map[doc_id]
        final_results.append({
            "id": doc_id,
            "rrf_score": rrf_scores[doc_id],
            "document": data['data']['document'],
            # Metadati per debug
            "semantic_rank": data['sem_rank'],
            "keyword_rank": data['kw_rank'],
        })
        
    return final_results

class ChatNode:
    """Rappresenta un singolo messaggio di chat recuperato."""
    def __init__(self, text: str, metadata: Dict[str, Any], id_: str, score: float = None):
        self.text = text
        self.metadata = metadata or {}
        self.node_id = id_
        self.score = score


class ChatStore:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        doc_prefix: str = CHAT_PREFIX,
        index_name: str = CHAT_INDEX_NAME,
        vector_field: str = "vector",
        vector_dim: int = DEFAULT_VECTOR_DIM,
        ollama_url: str = "http://localhost:11434/api/embeddings",
        ollama_model: str = "embeddinggemma:latest",
        decode_responses: bool = False,
    ):
        """Inizializza il ChatStore con i parametri Redis e di Embedding."""
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
    # Connessione e Indexing (Adattati da LoreStore)
    # -------------------------
    def connect(self) -> redis.Redis:
        """Crea e ritorna il client Redis, verificando la connessione."""
        if self.client:
            return self.client
        self.client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=self.decode_responses)
        try:
            self.client.ping()
        except Exception as e:
            logger.error("Impossibile connettersi a Redis: %s", e)
            raise
        logger.info("Connessione a Redis OK")
        return self.client

    def init_index(self, recreate: bool = False):
        """Crea l'indice RediSearch per i messaggi di chat (testo + vector)."""
        r = self.connect()

        if recreate:
            try:
                r.ft(self.index_name).dropindex(delete_documents=False)
                logger.info("Vecchio indice chat eliminato (recreate=True).")
            except Exception as e:
                logger.debug("dropindex: %s", e)

        # Schema più semplice per la chat memory
        schema = [
            TextField("text", weight=1.0),
            TagField("timestamp"), # Per filtri veloci e contesto
            VectorField(self.vector_field, "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": self.vector_dim,
                "DISTANCE_METRIC": "COSINE"
            })
        ]

        definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)
        try:
            r.ft(self.index_name).create_index(schema, definition=definition)
            logger.info("Indice chat creato: %s", self.index_name)
        except Exception as e:
            logger.warning("Creazione indice chat fallita (potrebbe già esistere): %s", e)

    def _doc_key(self, doc_id: str) -> str:
        return f"{self.doc_prefix}{doc_id}"

    @staticmethod
    def safe_decode(x: Any) -> Any:
        """Decodifica bytes->str se necessario."""
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8")
            except Exception:
                return x
        return x

    # -------------------------
    # Embedding (Copiato da LoreStore)
    # -------------------------
    def get_embedding(self, text: str) -> np.ndarray:
        """Richiede l'embedding al servizio configurato (Ollama di default)."""
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
    # Metodi di gestione (Adattati da chat_store.py originale)
    # -------------------------
    def count(self) -> int:
        """Ritorna il numero di messaggi indicizzati nella chat memory."""
        r = self.connect()
        try:
            info = r.ft(self.index_name).info()
            return int(info.get('num_docs', 0))
        except Exception:
            logger.exception("Error counting messages")
            return 0

    def create_id(self, user_message: str, assistant_response: str) -> str:
        """Genera un ID hash univoco per la conversazione."""
        hash_input = f"{user_message}{assistant_response}{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def create_metadata(self) -> Dict[str, Any]:
        """Crea i metadati essenziali per il messaggio (semplificato)."""
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_type": "dialogue",
        }

    def upsert(self, id_: str, document: str, metadata: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Indicizza un singolo messaggio di chat (documento) in Redis come Hash.
        """
        r = self.connect()
        key = self._doc_key(id_)

        try:
            # 1. Normalizza l'embedding in float32 bytes
            if isinstance(embedding, (list, tuple, np.ndarray)):
                vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
            elif isinstance(embedding, (bytes, bytearray)):
                vec_bytes = bytes(embedding)
            else:
                raise TypeError("Embedding deve essere list/np.ndarray o bytes")

            # 2. Prepara i campi Hash
            mapping = {
                "text": document.encode("utf-8"),
                # Metadati richiesti dallo schema RediSearch
                "timestamp": metadata.get("timestamp", "").encode("utf-8"),
                self.vector_field: vec_bytes
            }
            
            # Aggiungi tutti gli altri metadati come stringhe
            for k, v in metadata.items():
                if k not in ["timestamp"]:
                    mapping[k] = str(v).encode("utf-8")

            # 3. HSET (upsert)
            r.hset(key, mapping=mapping)
            logger.debug("Upserted chat message key=%s", key)
            return True
        except Exception:
            logger.exception("Error upserting to chat store")
            return False

    def retrieve_by_vector(self, query_text: str, top_k: int = 5) -> List[ChatNode]:
        """
        Ricerca vettoriale KNN nella memoria della chat.
        """
        r = self.connect()
        
        try:
            emb = self.get_embedding(query_text)
        except Exception:
            logger.error("Embedding fallito per la query di ricerca.")
            return []
            
        vec_bytes = emb.astype(np.float32).tobytes()

        # Query KNN: ritorna ID, testo, timestamp e il punteggio vettoriale (distanza)
        q = (
            Query(f"*=>[KNN {top_k} @{self.vector_field} $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("__key__", "text", "timestamp", "vector_score")
            .dialect(2)
        )

        try:
            res = r.ft(self.index_name).search(q, query_params={"vec": vec_bytes})
        except Exception:
            logger.exception("Chat index query failed")
            return []
            
        nodes = []
        for doc in res.docs:
            
            # Recupera l'ID del documento
            full_key = self.safe_decode(getattr(doc, "id"))
            doc_id = full_key.split(self.doc_prefix)[-1]
            
            # Per recuperare i metadati non inclusi nel RETURN FIELDS, usiamo HGETALL sulla chiave completa
            raw_data = r.hgetall(full_key)
            metadata = {
                 self.safe_decode(k): self.safe_decode(v)
                 for k, v in raw_data.items()
                 if self.safe_decode(k) not in ["text", self.vector_field]
            }

            distance = float(self.safe_decode(getattr(doc, "vector_score", 0)))
            
            nodes.append(ChatNode(
                text=self.safe_decode(getattr(doc, "text", "")),
                metadata=metadata,
                id_=doc_id,
                # Converto la distanza coseno (vicino a 0 è meglio) in punteggio di similarità (vicino a 1 è meglio)
                score=1.0 / (1.0 + distance) 
            ))
            
        logger.debug("retrieve_by_vector (chat) returning %d nodes", len(nodes))
        return nodes

    def delete_by_filter(self, where: Dict[str, Any]) -> bool:
        """
        Elimina i messaggi di chat tramite filtro.
        NOTA: RediSearch richiede una query per trovare le chiavi prima di poterle eliminare.
        """
        r = self.connect()
        
        # Costruiamo una query che cerca le chiavi per i filtri specificati
        filter_parts = []
        for key, value in where.items():
            filter_parts.append(f'@{key}:"{value}"')
        
        query_str = " ".join(filter_parts)
        
        if not query_str:
            logger.warning("Filtro vuoto, operazione di cancellazione annullata.")
            return False
            
        # Esegui la ricerca e ritorna solo gli ID delle chiavi
        try:
            res = r.ft(self.index_name).search(
                query_str, 
                query_params={},
                return_fields=["__key__"],
                # Paging alto per recuperare fino a 10000 chiavi
                params={"LIMIT": [0, 10000]} 
            )
        except Exception:
            logger.exception("Errore nella ricerca delle chiavi da eliminare.")
            return False

        keys_to_delete = [self.safe_decode(doc.__key__) for doc in res.docs]
        
        if not keys_to_delete:
            logger.info("Nessuna chiave trovata per il filtro, nessuna cancellazione eseguita.")
            return True
        
        # Elimina tutte le chiavi trovate
        try:
            deleted_count = r.delete(*keys_to_delete)
            logger.info(f"Eliminati {deleted_count} messaggi di chat con filtro {where}")
            return True
        except Exception:
            logger.exception("Errore durante l'eliminazione delle chiavi.")
            return False
        
    def keyword_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Esegue la ricerca full-text (keyword/lessicale) sulla chat history."""
        r = self.connect()
        
        # Definisci i campi da ritornare: usiamo l'alias $$SCORE
        return_fields = ["__key__", "text", "$$SCORE"] # <--- Ritorna a $$SCORE
        
        # Costruisci la Query: 
        query = (
            Query(query_text)
            .limit_fields("text")
            .paging(0, k)
            # RIMOZIONE DI .sort_by("score", asc=False)
            .return_fields(*return_fields) 
            .dialect(2)
        )
        
        logger.debug(f"Esecuzione ricerca testuale (k={k})...")
        
        # Esegui la ricerca
        results = r.ft(self.index_name).search(query)
        
        ranked_results = []
        # L'ordinamento è implicito per $$SCORE, l'iterazione fornisce il rank.
        for rank, doc in enumerate(results.docs): 
            
            full_key = self.safe_decode(getattr(doc, "id"))
            
            # NOTA: Per estrarre $$SCORE da un oggetto Document (che è un'istanza di dotdict),
            # usiamo doc.property o doc['property']. Qui usiamo doc['$$SCORE'].
            
            # Il punteggio è restituito come byte e va decodificato e convertito a float
            raw_score = self.safe_decode(getattr(doc, '$$SCORE', None))
            score = float(raw_score) if raw_score else 0.0
            
            ranked_results.append({
                "id": full_key.split(self.doc_prefix)[-1], 
                "rank": rank + 1,
                "document": self.safe_decode(getattr(doc, "text")),
                "score": score,
                "method": "keyword"
            })
        return ranked_results

    def hybrid_search(self, query_text: str, top_k_semantic: int = 5, top_k_keyword: int = 5, final_k: int = 5) -> List[Dict[str, Any]]:
        """
        Esegue Ricerca Ibrida (Semantic + Keyword) e fonde i risultati con RRF.
        
        Ritorna una lista di dizionari ordinati per punteggio RRF.
        """
        
        # 1. Ricerca Semantica (basata sul retrieve_by_vector esistente)
        # Nota: retrieve_by_vector ritorna ChatNode, qui abbiamo bisogno di un dict per l'RRF.
        # Converto il risultato di retrieve_by_vector in dict per la fusione.
        sem_nodes = self.retrieve_by_vector(query_text, top_k=top_k_semantic)
        sem_results = []
        for rank, node in enumerate(sem_nodes):
             sem_results.append({
                "id": node.node_id,
                "rank": rank + 1,
                "document": node.text,
                "score": node.score,
                "method": "semantic"
            })
            
        # 2. Ricerca Keyword
        key_results = self.keyword_search(query_text, k=top_k_keyword)
        
        logger.info(f"Chat Store: Semantica: {len(sem_results)} risultati. Keyword: {len(key_results)} risultati.")

        # 3. Fusione RRF
        if not sem_results and not key_results:
            return []
            
        fused_results = apply_rrf(sem_results, key_results)

        # 4. Ritorna solo i top_k finali
        return fused_results[:final_k]
# -------------------------
# Esempio di utilizzo (per il test)
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = ChatStore(vector_dim=768) # ADATTA vector_dim in base al tuo modello
    
    # 1. Connessione e creazione indice
    try:
        store.connect()
        store.init_index(recreate=True)
    except Exception:
        print("\n\nIMPOSSIBILE ESEGUIRE IL TEST. Assicurati che Redis Stack sia attivo su localhost:6379.")
        exit()

    # 2. Esempio di upsert
    test_id = store.create_id("user test message", "assistant test response")
    test_metadata = store.create_metadata()
    test_document = "Questa è la prima conversazione. Il mio personaggio preferito è Aethel."
    
    # Calcola l'embedding (qui devi usare lo stesso modello dell'indicizzatore)
    try:
        # Nota: Ollama get_embedding è sincrono e potrebbe impiegare tempo
        test_embedding = store.get_embedding(test_document).tolist()
    except Exception:
        print("Errore nel calcolo dell'embedding. Assicurati che Ollama sia in esecuzione.")
        exit()

    store.upsert(test_id, test_document, test_metadata, test_embedding)
    print(f"\nMessaggi in store: {store.count()}")

    # 3. Esempio di retrieve IBRIDO
    q_hybrid = "dimmi qualcosa sui minerali che ho menzionato prima"
    results_hybrid = store.hybrid_search(q_hybrid, top_k_semantic=5, top_k_keyword=5, final_k=3)
    print("\nRisultati di Ricerca Ibrida (RRF) sulla Chat History:")
    for r in results_hybrid:
        print(f"- id={r['id'][:8]}, rrf_score={r['rrf_score']:.6f}")
        print(f"  testo: {r['document']}")
        print(f"  Rank Semantico: {r['semantic_rank'] if r['semantic_rank'] else 'N/A'}, Rank Keyword: {r['keyword_rank'] if r['keyword_rank'] else 'N/A'}")
        
    # 4. Esempio di delete
    # store.delete_by_filter(where={"timestamp": test_metadata["timestamp"]})
    # print(f"\nMessaggi dopo la cancellazione: {store.count()}")