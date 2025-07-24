import os
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from tqdm import tqdm  # Import tqdm for progress bar

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PERSIST_DIR = "storage"  # Define a constant for the persist directory

def create_vector_store(docs_path: str = "docs"):
    """
    Creates or updates a vector store from PDF documents in the specified directory.

    Args:
        docs_path (str): The path to the directory containing the PDF documents.
                         Defaults to "docs".

    Returns:
        llama_index.core.VectorStoreIndex: The created or updated vector store index.
    """

    # Check if the documents directory exists
    if not os.path.exists(docs_path):
        raise ValueError(f"Documents directory '{docs_path}' not found.")

    # Load the documents from the directory
    logging.info(f"Loading documents from {docs_path}")
    documents = SimpleDirectoryReader(docs_path).load_data()

    # Initialize the LLM
    logging.info("Initializing LLM (Ollama)")
    llm = Ollama(model="deepseek-r1:1.5b")
    Settings.llm = llm

    # Initialize the embeddings model
    logging.info("Initializing embedding model (Ollama)")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text:v1.5")
    Settings.embed_model = embed_model

    # Check if the vector store already exists
    if os.path.exists(PERSIST_DIR):
        # Load the existing index from disk
        logging.info("Loading existing vector store index from disk")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        logging.info("Existing vector store index loaded successfully.")

        # Insert new documents into the existing index
        logging.info("Inserting new documents into the existing index")
        for doc in tqdm(documents[:5], desc="Inserting Documents"): 
            index.insert(doc)
        logging.info("New documents inserted successfully.")
    else:
        # Create the vector store index
        logging.info("Creating new vector store index")
        index = VectorStoreIndex.from_documents(tqdm(documents[:5], desc="Creating Vector Store")) # documents[1:10]
        logging.info("Vector store index created successfully.")

    return index

if __name__ == "__main__":
    try:
        # Create or update the vector store
        vector_store_index = create_vector_store()

        # Save the vector store to disk
        vector_store_index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Vector store created/updated and persisted to '{PERSIST_DIR}' directory.")

    except Exception as e:
        print(f"Error creating/updating vector store: {e}")