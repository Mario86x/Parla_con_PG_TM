import chromadb
import logging
import time
from typing import List, Dict, Any
from BM25_retriever import bm25_search

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LoreStore")

LORE_PERSIST_DIR = "chroma_db"
LORE_COLLECTION = "ardania_lore"
MAX_BM25_DOCS = 1000

class SimpleNode:
    def __init__(self, text: str, metadata: Dict[str, Any], id_: str, score: float = None):
        self.text = text
        self.metadata = metadata or {}
        self.node_id = id_
        self.score = score

class LoreStore:
    def __init__(self, persist_dir: str = LORE_PERSIST_DIR, collection_name: str = LORE_COLLECTION):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded lore collection '{collection_name}' from '{persist_dir}'")
        except Exception:
            logger.exception("Lore collection not found")
            raise RuntimeError(f"Lore collection '{collection_name}' not found in '{persist_dir}'")

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            logger.exception("Error counting documents")
            return 0

    def sample(self, limit: int = 5) -> List[str]:
        """Get sample documents"""
        try:
            res = self.collection.get(limit=limit, include=["documents", "metadatas"])
            return res.get("documents", [[]])[0]
        except Exception:
            logger.exception("Error sampling documents")
            return []

    def retrieve_by_vector(self, embedding: List[float], top_k: int = 5) -> List[SimpleNode]:
        if not embedding:
            logger.warning("Empty embedding provided to retrieve_by_vector")
            return []
        start = time.time()
        logger.debug("Starting Chroma vector query (top_k=%d)...", top_k)
        try:
            res = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            logger.exception("Chroma query failed")
            return []
        elapsed = time.time() - start
        logger.debug("Chroma vector query done in %.3fs", elapsed)

        nodes = []
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            id_ = f"lore_{i}"
            dist = dists[i] if i < len(dists) else None
            score = None
            if dist is not None:
                try:
                    score = 1.0 / (1.0 + float(dist))
                except Exception:
                    score = None
            nodes.append(SimpleNode(text=doc, metadata=meta, id_=id_, score=score))

        logger.debug("retrieve_by_vector returning %d nodes", len(nodes))
        return nodes

    def retrieve_by_bm25(self, query: str, top_k: int = 5) -> List[SimpleNode]:
        logger.debug("Starting BM25 retrieval (top_k=%d)...", top_k)
        try:
            # Cap number of documents to avoid huge in-memory BM25 indexing
            count = None
            try:
                count = self.collection.count()
            except Exception:
                logger.debug("Could not get collection count for BM25; proceeding without limit")

            if count and count > MAX_BM25_DOCS:
                limit = MAX_BM25_DOCS
                all_docs_res = self.collection.get(limit=limit, include=["documents", "metadatas"])
            else:
                all_docs_res = self.collection.get(include=["documents", "metadatas"])

            all_docs = all_docs_res.get("documents", [[]])[0]
            all_metadatas = all_docs_res.get("metadatas", [[]])[0]

            if not all_docs:
                logger.warning("No documents available for BM25 search")
                return []

            bm25_results = bm25_search(query, all_docs, top_k=top_k, lang='italian')
            nodes: List[SimpleNode] = []
            for res in bm25_results:
                idx = res["index"]
                if idx < len(all_docs):
                    nodes.append(SimpleNode(
                        text=res["document"],
                        metadata=all_metadatas[idx] if idx < len(all_metadatas) else {},
                        id_=f"bm25_{idx}",
                        score=res["score"]
                    ))

            logger.debug("retrieve_by_bm25 returning %d nodes", len(nodes))
            return nodes
        except Exception:
            logger.exception("BM25 search failed")
            return []