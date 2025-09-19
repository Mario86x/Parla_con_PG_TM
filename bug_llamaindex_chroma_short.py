import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

# Create embedding model
embed_model = OllamaEmbedding(model_name="embeddinggemma")
Settings.embed_model = embed_model

# Create Chroma collection
PERSIST_DIR = "chromadb_bug"
COLLECTION_NAME = "vector_store_test8"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# Add documents
for i in range(5):
    doc = f"This is document {i}"
    embedding = embed_model.get_text_embedding(doc)
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"doc_{i}"],
    )

# Retrieval with ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=3, embed_model=embed_model)

query_text = "get document 1"
nodes = retriever.retrieve(query_text)

for node in nodes:
    print(f"Score: {node.score:.4f} | Content: {node.text}")


# ------------------------------
# Direct retrieval from ChromaDB
# ------------------------------
query_embedding = embed_model.get_text_embedding(query_text)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "distances"]
)

print(f"\n[Direct ChromaDB] Top 3 similar documents to '{query_text}':\n")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"Distance: {dist:.4f} | Content: {doc}")