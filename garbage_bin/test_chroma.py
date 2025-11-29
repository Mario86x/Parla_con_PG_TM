import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import chromadb

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_chroma")

LORE_PERSIST_DIR = "chroma_db"
LORE_COLLECTION = "ardania_lore"

def run_with_timeout(fn, timeout=10.0):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            logger.warning("Operation timed out after %.1fs", timeout)
            return None
        except Exception:
            logger.exception("Operation raised")
            return None

def main():
    logger.info("Opening Chroma client at %s", LORE_PERSIST_DIR)
    try:
        client = chromadb.PersistentClient(path=LORE_PERSIST_DIR)
    except Exception:
        logger.exception("Failed to create PersistentClient")
        return

    try:
        coll = client.get_collection(name=LORE_COLLECTION)
    except Exception:
        logger.exception("Failed to get collection '%s'", LORE_COLLECTION)
        return

    # 1) count
    def do_count():
        return coll.count()
    t0 = time.time()
    cnt = run_with_timeout(do_count, timeout=5.0)
    logger.info("count() -> %s (t=%.3fs)", repr(cnt), time.time() - t0)

    # 2) sample get (do not request 'distances' here)
    def do_get():
        return coll.get(limit=3, include=["documents", "metadatas"])
    t0 = time.time()
    sample = run_with_timeout(do_get, timeout=5.0)
    logger.info("get() -> type=%s (t=%.3fs)", type(sample), time.time() - t0)
    if sample:
        logger.debug("sample keys: %s", list(sample.keys()))
        docs_field = sample.get("documents")
        if isinstance(docs_field, list) and len(docs_field) > 0:
            docs = docs_field[0]
        else:
            docs = docs_field or []
        logger.info("sample documents returned: %d", len(docs))

    # 3) vector query (use zero vector with same length as your embeddings)
    emb_len = 768  # adjust if your embeddings have different length
    zero_emb = [0.0] * emb_len
    def do_query():
        return coll.query(query_embeddings=[zero_emb], n_results=1, include=["documents", "metadatas", "distances"])
    t0 = time.time()
    qry = run_with_timeout(do_query, timeout=10.0)
    logger.info("query() -> %s (t=%.3fs)", "None" if qry is None else "OK", time.time() - t0)
    if qry:
        logger.debug("query keys: %s", list(qry.keys()))
        docs = qry.get("documents", [[]])[0]
        logger.info("query returned docs=%d", len(docs))

if __name__ == "__main__":
    main()