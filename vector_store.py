import os
from dotenv import load_dotenv
import logging
import sys
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document
)
from llama_parse import LlamaParse
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llm import init_llm, init_local_embed_model
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PERSIST_DIR = "chroma_storage"
load_dotenv()

def create_vector_store(docs_path: str = "docs"):
    """
    Crea o aggiorna un vector store locale con Chroma.
    """

    if not os.path.exists(docs_path):
        raise ValueError(f"Documents directory '{docs_path}' not found.")

    logging.info(f"Loading documents from {docs_path}")
    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        result_type="markdown",
        verbose=True,
    )
    file_extractor = {".pdf": parser}



    documents = SimpleDirectoryReader(
        docs_path, file_extractor=file_extractor
    ).load_data()

    # Aggiungiamo un ID sequenziale ai chunk come metadata
    enriched_docs = []
    for doc_id, doc in enumerate(documents):
        for i, node in enumerate(doc.get_nodes()):
            node.metadata["doc_id"] = str(doc_id)
            node.metadata["seq"] = i
        enriched_docs.append(doc)

    # Inizializzo LLM ed embedding model
    logging.info("Initializing LLM (Google GenAI)")
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = init_llm(api_key)
    Settings.llm = llm

    logging.info("Initializing embedding model (Google GenAI)")
    # embed_model = init_embed_model(api_key)
    embed_model = init_local_embed_model()
    Settings.embed_model = embed_model

    # Creo client Chroma con persistenza su disco
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("docs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logging.info("Creating/updating Chroma index")
    index = VectorStoreIndex.from_documents(
        tqdm(enriched_docs, desc="Creating Chroma Vector Store"),
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=100
    )

    return index


def query_with_context(index, query: str, top_k: int = 2):
    """
    Query che ritorna i chunk rilevanti insieme a previous e next (in ordine narrativo).
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    output = []
    for res in results:
        node = res.node
        seq = node.metadata.get("seq")
        doc_id = node.metadata.get("doc_id")

        # Prev
        prev_nodes = index.vector_store.query(
            query_embedding=None,
            n_results=1,
            where={"doc_id": doc_id, "seq": seq - 1}
        ) if seq > 0 else None

        if prev_nodes and len(prev_nodes["documents"]) > 0:
            output.append({
                "type": "previous",
                "text": prev_nodes["documents"][0]
            })

        # Match
        output.append({
            "type": "match",
            "text": node.get_content()
        })

        # Next
        next_nodes = index.vector_store.query(
            query_embedding=None,
            n_results=1,
            where={"doc_id": doc_id, "seq": seq + 1}
        )
        if next_nodes and len(next_nodes["documents"]) > 0:
            output.append({
                "type": "next",
                "text": next_nodes["documents"][0]
            })

    return output


if __name__ == "__main__":
    try:
        index = create_vector_store()
        print(f"Vector store creato/aggiornato in '{PERSIST_DIR}'.")

        # Query di test
        query = "chi sono i goblin?"
        results = query_with_context(index, query, top_k=2)

        for r in results:
            print(f"[{r['type'].upper()}] {r['text'][:200]}...")

    except Exception as e:
        print(f"Error: {e}")
