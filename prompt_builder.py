import logging
from typing import List, Dict, Any
from templates import SYSTEM_PROMPT, CHARACTER_PROMPT

logging.basicConfig(level=logging.INFO)

# La lista lore_nodes usa Dict[str, Any] (da LoreStore.hybrid_search)
# La lista chat_nodes usa List[ChatNode] (da ChatStore.retrieve_by_vector o HybridRetriever.retrieve_chat_hybrid)
# Non possiamo usare un Type Hint corretto qui senza importare ChatNode, ma possiamo gestire i due tipi.
def build_prompt(
    running_story: str,
    lore_nodes: List[Dict],
    chat_nodes: List[Any], # Usiamo Any per accettare sia Dict che ChatNode
    user_message: str,
    max_history: int = 10
) -> str:
    """Build prompt combining all context sources"""
    try:
        # Format lore context (Assume Dict type)
        lore_text = ""
        if lore_nodes:
            lore_parts = []
            for node in lore_nodes:
                # Lore usa .get() perché restituisce Dict
                heading = node.get("headings", "Unknown") 
                text = node.get("source_text", "") or node.get("text", "") # Usa source_text/text
                text = text[:500] 
                lore_parts.append(f"[LORE] {heading}: {text}")
            lore_text = "\n".join(lore_parts)

        # Format chat context (Assume ChatNode type)
        chat_text = ""
        if chat_nodes:
            chat_parts = []
            for node in chat_nodes[-max_history:]:
                # CORREZIONE: Usa l'attributo .text se è un oggetto ChatNode
                # Se fosse un Dict, .get() è corretto. Gestiamo ChatNode:
                if hasattr(node, 'text'):
                    text = node.text[:300] # Accesso diretto all'attributo .text
                else:
                    # Fallback nel caso in cui stia ricevendo un Dict (es. da un'altra parte del codice)
                    text = node.get("text", "")[:300]
                
                chat_parts.append(f"[CHAT] {text}")
            chat_text = "\n".join(chat_parts)

        # Build final prompt
        # ... (il resto della funzione è omesso per brevità, ma non deve essere modificato) ...
        prompt = f"""{SYSTEM_PROMPT.template}

{CHARACTER_PROMPT.template}

## CONTESTO DELLA LORE:
{lore_text if lore_text else "[Nessun contesto di lore disponibile]"}

## CRONOLOGIA DELLA CHAT:
{chat_text if chat_text else "[Nessuna cronologia disponibile]"}

## STORIA IN CORSO:
{running_story if running_story else "[Inizio della conversazione]"}

## MESSAGGIO DEL GIOCATORE:
{user_message}

## IL TUO PERSONAGGIO:"""

        logging.debug("Prompt built successfully")
        return prompt
    except Exception as e:
        logging.error(f"Error building prompt: {e}")
        return ""