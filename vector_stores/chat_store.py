import chromadb
import hashlib
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ChatStore")

CHAT_PERSIST_DIR = "chroma_db"
CHAT_COLLECTION = "ardania_chat_memory"

class ChatNode:
    def __init__(self, text: str, metadata: Dict[str, Any], id_: str, score: float = None):
        self.text = text
        self.metadata = metadata or {}
        self.node_id = id_
        self.score = score

class ChatStore:
    def __init__(self, persist_dir: str = CHAT_PERSIST_DIR, collection_name: str = CHAT_COLLECTION):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Chat memory storage for conversations"}
            )
            logger.info(f"Chat collection '{collection_name}' ready in '{persist_dir}'")
        except Exception:
            logger.exception("Error initializing chat collection")
            raise

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            logger.exception("Error counting messages")
            return 0

    def create_id(self, user_message: str, assistant_response: str) -> str:
        hash_input = f"{user_message}{assistant_response}{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def create_metadata(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_type": "regular_chat",
            "turn_number": self.count() + 1,
            "message_type": "dialogue",
            "importance_score": 0.5
        }

    def upsert(self, id_: str, document: str, metadata: Dict[str, Any], embedding: List[float]) -> bool:
        try:
            self.collection.upsert(
                ids=[id_],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug("Upserted chat message id=%s", id_[:8])
            return True
        except Exception:
            logger.exception("Error upserting to chat store")
            return False

    def retrieve_by_vector(self, embedding: List[float], top_k: int = 5) -> List[ChatNode]:
        if not embedding:
            logger.warning("Empty embedding provided to retrieve_by_vector")
            return []
        start = time.time()
        logger.debug("Starting Chat Chroma query (top_k=%d)...", top_k)
        try:
            res = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            logger.exception("Chat collection query failed")
            return []
        elapsed = time.time() - start
        logger.debug("Chat Chroma query done in %.3fs", elapsed)

        nodes = []
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            id_ = f"chat_{i}"
            dist = dists[i] if i < len(dists) else None
            score = None
            if dist is not None:
                try:
                    score = 1.0 / (1.0 + float(dist))
                except Exception:
                    score = None
            nodes.append(ChatNode(text=doc, metadata=meta, id_=id_, score=score))
        logger.debug("retrieve_by_vector (chat) returning %d nodes", len(nodes))
        return nodes

    def delete_by_filter(self, where: Dict[str, Any]) -> bool:
        try:
            self.collection.delete(where=where)
            logger.info("Deleted chat messages with filter %s", where)
            return True
        except Exception:
            logger.exception("Error deleting from chat store")
            return False