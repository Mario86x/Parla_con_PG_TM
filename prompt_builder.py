import logging
from typing import List, Dict
from templates import SYSTEM_PROMPT, CHARACTER_PROMPT

logging.basicConfig(level=logging.INFO)

def build_prompt(
    running_story: str,
    lore_nodes: List[Dict],
    chat_nodes: List[Dict],
    user_message: str,
    max_history: int = 10
) -> str:
    """Build prompt combining all context sources"""
    try:
        # Format lore context
        lore_text = ""
        if lore_nodes:
            lore_parts = []
            for node in lore_nodes:
                heading = node.get("metadata", {}).get("headings", "Unknown")
                text = node.get("text", "")[:500]  # Truncate to 500 chars
                lore_parts.append(f"[LORE] {heading}: {text}")
            lore_text = "\n".join(lore_parts)

        # Format chat context (last max_history messages)
        chat_text = ""
        if chat_nodes:
            chat_parts = []
            for node in chat_nodes[-max_history:]:
                text = node.get("text", "")[:300]  # Truncate to 300 chars
                chat_parts.append(f"[CHAT] {text}")
            chat_text = "\n".join(chat_parts)

        # Build final prompt
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