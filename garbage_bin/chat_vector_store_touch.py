import chromadb
import logging

# Constants for chat vector store
PERSIST_DIR = "chroma_db"
CHAT_COLLECTION = "ardania_chat_memory"

def init_chat_collection():
    """
    Initialize empty ChromaDB collection for chat memory.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        collection = chroma_client.get_or_create_collection(
            name=CHAT_COLLECTION,
            metadata={
                "description": "Memoria delle conversazioni con il personaggio",
                "hnsw:space": "cosine"
            }
        )
        logging.info(f"Chat collection '{CHAT_COLLECTION}' ready in '{PERSIST_DIR}'")
        return collection
    except Exception as e:
        logging.error(f"Error initializing chat collection: {e}")
        raise

def main():
    """Initialize empty chat vector store"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    collection = init_chat_collection()
    count = collection.count()
    logging.info(f"Chat collection initialized with {count} documents")

    # Example commands to clear/filter collection (commented out)
    """
    # Delete all regular chat memories older than X
    collection.delete(
        where={
            "memory_type": "regular_chat",
            "timestamp": {"$lt": "2024-03-15T00:00:00"}
        }
    )

    # Delete all memories except pinned ones
    collection.delete(
        where={
            "memory_type": {"$ne": "pinned_memory"}
        }
    )

    # Delete memories with low importance score
    collection.delete(
        where={
            "importance_score": {"$lt": 0.5},
            "memory_type": "regular_chat"
        }
    )

    # Delete all memories for specific user
    collection.delete(
        where={
            "user_id": "specific_user_id"
        }
    )

    # Delete entire collection
    # chroma_client.delete_collection(CHAT_COLLECTION)
    """

if __name__ == "__main__":
    main()