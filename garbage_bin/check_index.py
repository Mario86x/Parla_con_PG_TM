import os, traceback, json
from llama_index.core import StorageContext, load_index_from_storage

PERSIST_DIRS = ["storage", "chroma_db"]
print("PWD:", os.getcwd())
for d in PERSIST_DIRS:
    print(f"\n--- {d} exists:", os.path.exists(d))
    if os.path.exists(d):
        print(" listing:", os.listdir(d)[:20])

# try loading llama-index from 'storage'
for d in PERSIST_DIRS:
    try:
        print(f"\nTrying StorageContext.from_defaults(persist_dir='{d}')")
        sc = StorageContext.from_defaults(persist_dir=d)
        idx = load_index_from_storage(sc)
        print("Loaded index from", d)
        try:
            docs = list(idx.docstore.docs.keys())[:5]
            print("docstore sample keys:", docs)
        except Exception as e:
            print("Could not read docstore keys:", e)
    except Exception as e:
        print("Failed loading from", d)
        traceback.print_exc()