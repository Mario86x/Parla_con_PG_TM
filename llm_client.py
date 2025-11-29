import logging
import time
from typing import List
from llm import init_llm, init_local_embed_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("llm_client")

def init_clients(api_key: str = None):
    try:
        llm = init_llm(api_key)
        embed = init_local_embed_model()
        logger.info("LLM and embedding model initialized")
        return llm, embed
    except Exception as e:
        logger.exception("Error initializing clients")
        raise

def embed_text(embed_model, text: str) -> List[float]:
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided to embed_text")
            return []
        t0 = time.time()
        embedding = embed_model.get_text_embedding(text)
        logger.debug("embed_text done in %.3fs (len=%s)", time.time() - t0, len(embedding) if hasattr(embedding, '__len__') else "unknown")
        return embedding
    except Exception as e:
        logger.exception("Error generating embedding")
        return []

def generate_text(llm, prompt: str) -> str:
    try:
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to generate_text")
            return ""
        t0 = time.time()
        response = llm.complete(prompt)
        logger.debug("generate_text done in %.3fs", time.time() - t0)
        return response.text.strip()
    except Exception as e:
        logger.exception("Error generating text from LLM")
        return ""