import logging
import time
from typing import List, Dict, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    """Combine vector similarity + BM25 retrieval (sequential, no parallelism)"""
    
    def __init__(self, lore_store, chat_store):
        self.lore_store = lore_store
        self.chat_store = chat_store

    def retrieve_lore_hybrid(self, query: str, embedding: List[float], top_k: int = 5, alpha: float = 0.6) -> List[Dict]:
        """
        Retrieve from lore using both vector similarity and BM25 sequentially.
        Vector results are weighted by `alpha`, BM25 by (1-alpha).
        """
        start = time.time()
        try:
            logger.debug("HybridRetriever: starting vector retrieval (top_k=%d)", top_k)
            try:
                vector_nodes = self.lore_store.retrieve_by_vector(embedding, top_k=top_k)
            except Exception:
                logger.exception("Vector retrieval failed")
                vector_nodes = []
            logger.debug("HybridRetriever: vector retrieval done (found=%d) in %.3fs", len(vector_nodes), time.time() - start)

            logger.debug("HybridRetriever: starting BM25 retrieval (top_k=%d)", top_k)
            try:
                bm25_nodes = self.lore_store.retrieve_by_bm25(query, top_k=top_k)
            except Exception:
                logger.exception("BM25 retrieval failed")
                bm25_nodes = []
            logger.debug("HybridRetriever: BM25 retrieval done (found=%d) in %.3fs", len(bm25_nodes), time.time() - start)

            vector_results = [
                {
                    "text": getattr(n, "text", ""),
                    "metadata": getattr(n, "metadata", {}),
                    "score": (getattr(n, "score", None) or 0.0),
                    "source": "vector",
                    "node_id": getattr(n, "node_id", None)
                }
                for n in vector_nodes
            ]
            bm25_results = [
                {
                    "text": getattr(n, "text", ""),
                    "metadata": getattr(n, "metadata", {}),
                    "score": (getattr(n, "score", None) or 0.0),
                    "source": "bm25",
                    "node_id": getattr(n, "node_id", None)
                }
                for n in bm25_nodes
            ]

            # Combine and deduplicate by text prefix
            combined: Dict[str, Dict[str, Any]] = {}
            for res in vector_results:
                key = (res["text"] or "")[:200]
                combined[key] = {**res, "combined_score": res["score"] * alpha}

            for res in bm25_results:
                key = (res["text"] or "")[:200]
                if key in combined:
                    combined[key]["combined_score"] += res["score"] * (1 - alpha)
                    combined[key]["source"] = "hybrid"
                else:
                    combined[key] = {**res, "combined_score": res["score"] * (1 - alpha)}

            results = sorted(combined.values(), key=lambda x: x.get("combined_score", 0.0), reverse=True)[:top_k]
            logger.debug("HybridRetriever: combined results=%d total_time=%.3fs", len(results), time.time() - start)
            return results

        except Exception:
            logger.exception("Error in hybrid retrieval")
            return []

    def retrieve_chat_vector(self, embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Retrieve from chat memory using vector similarity only"""
        try:
            logger.debug("HybridRetriever: retrieving chat vector (top_k=%d)", top_k)
            chat_nodes = self.chat_store.retrieve_by_vector(embedding, top_k=top_k)
            results = [
                {
                    "text": getattr(n, "text", ""),
                    "metadata": getattr(n, "metadata", {}),
                    "score": (getattr(n, "score", None) or 0.0),
                    "source": "chat_vector",
                    "node_id": getattr(n, "node_id", None)
                }
                for n in chat_nodes
            ]
            logger.debug("HybridRetriever: chat vector results=%d", len(results))
            return results
        except Exception:
            logger.exception("Error retrieving from chat")
            return []