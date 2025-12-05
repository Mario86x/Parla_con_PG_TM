from typing import List, Any, Dict
import logging
# Importa le classi store:
from vector_stores.lore_store import LoreStore
# Assumo che l'implementazione Redis/RRF del ChatStore sia stata aggiornata per l'RRF.
from vector_stores.chat_store import ChatStore
from vector_stores.chat_store import ChatNode # Importa il nodo per i risultati

logger = logging.getLogger("HybridRetriever")

# Funzione ausiliaria per mappare i risultati RRF (Dict) in ChatNode
# in modo da mantenere la coerenza con il resto del workflow.
def _map_chat_dict_to_node(chat_result: Dict[str, Any]) -> ChatNode:
    """Mappa il dizionario risultato da ChatStore.hybrid_search (RRF) a ChatNode."""
    # Usa rrf_score come punteggio primario
    score = chat_result.get('rrf_score')
    
    # Costruisce i metadati
    metadata = {
        'semantic_rank': chat_result.get('semantic_rank'),
        'keyword_rank': chat_result.get('keyword_rank'),
        # Puoi aggiungere altri campi qui se ritornati dalla ricerca RRF
    }
    
    # NOTA: Per un ChatStore basato su Redis, il 'document' qui è il campo 'text'
    return ChatNode(
        text=chat_result.get('document', ''),
        metadata=metadata,
        id_=chat_result.get('id', 'unknown'),
        score=score
    )

# ----------------------------------------------------------------------
# CLASSE PRINCIPALE
# ----------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestra le operazioni di ricerca ibrida tra LoreStore (documenti) e ChatStore (memoria conversazionale).
    """
    def __init__(self, lore_store: LoreStore, chat_store: ChatStore):
        self.lore = lore_store
        self.chat = chat_store 
        logger.info("HybridRetriever initialized.")

    def retrieve_lore_hybrid(self, query: str, embedding: List[float], top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Esegue la ricerca ibrida (RRF) nella base di conoscenza (LoreStore).
        
        Chiama il metodo hybrid_search di LoreStore.
        LoreStore.hybrid_search gestisce internamente la generazione dell'embedding della query.
        """
        logger.debug("Lore hybrid search with final_k=%d", top_k)
        # La LoreStore usa già il metodo hybrid_search (RRF)
        return self.lore.hybrid_search(
            query_text=query,
            top_k_semantic=int(top_k * 1.5), # cerca più risultati iniziali per la fusione
            top_k_keyword=int(top_k * 2),    
            final_k=top_k
        )

    def retrieve_chat_hybrid(self, query: str, embedding: List[float], top_k: int = 3, alpha: float = 0.5) -> List[ChatNode]:
        """
        Esegue la ricerca ibrida (RRF) nella memoria della chat (ChatStore Redis).
        
        Ritorna una lista di ChatNode.
        """
        logger.debug("Chat hybrid search with final_k=%d", top_k)
        
        # Chiama il metodo hybrid_search del ChatStore (versione Redis/RRF)
        raw_results = self.chat.hybrid_search(
            query_text=query,
            top_k_semantic=int(top_k * 1.5),
            top_k_keyword=int(top_k * 2),
            final_k=top_k
        )
        
        # Mappa i risultati grezzi (dizionari) in ChatNode per coerenza col workflow
        chat_nodes = [_map_chat_dict_to_node(res) for res in raw_results]
        return chat_nodes