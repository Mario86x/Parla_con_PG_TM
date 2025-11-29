import time
from vector_stores.lore_store import LoreStore

l = LoreStore()
emb = [0.0] * 768

t0 = time.time()
vec = l.retrieve_by_vector(emb, top_k=3)
print("vector done:", len(vec), "took", time.time()-t0)

t0 = time.time()
bm = l.retrieve_by_bm25("salve chi sei", top_k=3)
print("bm25 done:", len(bm), "took", time.time()-t0)