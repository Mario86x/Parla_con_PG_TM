from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llm import init_llm, init_local_embed_model
from templates import SYSTEM_PROMPT, CHARACTER_PROMPT
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
import os
import tiktoken
import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from datetime import datetime
import hashlib

LORE_PERSIST_DIR = "chroma_db"
LORE_COLLECTION = "ardania_lore"
CHAT_PERSIST_DIR = "chroma_db"
CHAT_COLLECTION = "ardania_chat_memory"

class UserMessageEvent(Event):
    message: str
    verbose: bool = False  # Flag to indicate if debug info should be printed

class AssistantResponseEvent(Event):
    response: str

class ChatWorkflow(Workflow):
    def __init__(self, api_key: str):
        super().__init__()
        self.llm = init_llm(api_key)
        # self.embed_model = OllamaEmbedding(model_name="nomic-embed-text:v1.5")
        self.embed_model = init_local_embed_model()
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Load both vector stores
        self.lore_index = self._load_lore_store()
        self.chat_index = self._load_chat_store()
        print("Chat workflow initialized with LLM and vector store.")

    def _load_lore_store(self):
        """Load the existing lore vector store"""
        chroma_client = chromadb.PersistentClient(path=LORE_PERSIST_DIR)
        try:
            lore_collection = chroma_client.get_collection(name=LORE_COLLECTION)
            print(f"Lore collection loaded with {lore_collection.count()} documents")
            vector_store = ChromaVectorStore(chroma_collection=lore_collection)
            return VectorStoreIndex.from_vector_store(vector_store=vector_store, store_nodes_override=True)
        except Exception as e:
            logging.error(f"Error loading lore store: {e}")
            raise

    def _load_chat_store(self):
        """Load or create chat memory vector store"""
        chroma_client = chromadb.PersistentClient(path=CHAT_PERSIST_DIR)
        try:
            chat_collection = chroma_client.get_or_create_collection(
                name=CHAT_COLLECTION,
                metadata={"description": "Dynamic chat memory"}
            )
            print(f"Chat memory collection ready with {chat_collection.count()} messages")
            vector_store = ChromaVectorStore(chroma_collection=chat_collection)
            return VectorStoreIndex.from_vector_store(vector_store=vector_store, store_nodes_override=True)
        except Exception as e:
            logging.error(f"Error with chat memory store: {e}")
            raise

    async def _update_running_story(self, ctx: Context, new_content: str):
        running_story = await ctx.store.get("running_story", "")
        running_story += f"\n\n{new_content}"
        await ctx.store.set("running_story", running_story)
    
    def _create_chat_metadata(self, user_message: str, assistant_response: str) -> dict[str, any]:
        """Create metadata for a chat exchange"""
        # Create unique hash from message + timestamp for deduplication
        hash_input = f"{user_message}{assistant_response}{datetime.now().isoformat()}"
        hash_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        
        return {
            "hash": hash_id,
            "timestamp": datetime.now().isoformat(),
            "memory_type": "chat",
            "turn_number": len(self.chat_index.docstore.docs) + 1,
            "message_type": "dialogue",
            "user_message": user_message,
            "assistant_response": assistant_response,
            "importance_score": 0.5  # Default score, can be updated later
        }

    async def _save_to_chat_memory(self, user_message: str, assistant_response: str):
        """Save a conversation exchange to chat memory using optimized approach"""
        try:
            # Create the chat content
            exchange = f"""User Message: {user_message}\n
                         Assistant Response: {assistant_response}"""
            
            # Generate metadata
            metadata = self._create_chat_metadata(user_message, assistant_response)
            
            # Get collection directly for batch operations
            chroma_client = chromadb.PersistentClient(path=CHAT_PERSIST_DIR)
            chat_collection = chroma_client.get_collection(name=CHAT_COLLECTION)
            
            # Calculate embedding using the same model as vector store
            embedding = self.embed_model.get_text_embedding(exchange)
            
            # Upsert into collection
            chat_collection.upsert(
                ids=[metadata["hash"]],
                documents=[exchange],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            logging.info(f"Saved chat memory with hash {metadata['hash'][:8]}")
            
        except Exception as e:
            logging.error(f"Error saving to chat memory: {e}")
            raise

    @step
    async def start_chat(self, ctx: Context, ev: StartEvent) -> UserMessageEvent:
        print("Welcome to the chat! Let's create a story together.")
        await self._update_running_story(ctx, "")
        return UserMessageEvent(message="Start chat")

    @step
    async def get_user_message(self, ctx: Context, ev: AssistantResponseEvent) -> UserMessageEvent:
        print("\nPrevious response:")
        print(ev.response)
        verbose = False
        user_message = input("\nYour message: ")
        if "<debug>" in user_message:
            user_message = input("\nYour message (debug): ")
            verbose = True
        await self._update_running_story(ctx, f"\nUser: {user_message}")
        return UserMessageEvent(message=user_message, verbose=verbose)

    @step
    async def generate_response(self, ctx: Context, ev: UserMessageEvent) -> AssistantResponseEvent | StopEvent:
        running_story = await ctx.store.get("running_story", "")
        # last 10 messages
        last_10_messages = "\n".join(running_story.split("\n")[-20:])

        # Parallel retrieval from both stores
        lore_retriever = self.lore_index.as_retriever(similarity_top_k=10)
        chat_retriever = self.chat_index.as_retriever(similarity_top_k=5)

        lore_bm25_retriever = BM25Retriever.from_defaults(index=self.lore_index,
                                                          language='en',
                                                          similarity_top_k=10)
        chat_bm25_retriever = BM25Retriever.from_defaults(index=self.chat_index,
                                                          language='en',
                                                          similarity_top_k=5)


        # Get relevant nodes from both stores
        lore_nodes = lore_retriever.retrieve(ev.message)
        context_nodes=lore_retriever.retrieve(last_10_messages)
        lore_nodes.extend(context_nodes)
        lore_nodes_bm25 = lore_bm25_retriever.retrieve(ev.message)
        lore_nodes.extend(lore_nodes_bm25)
        
        # deduplicate lore nodes
        lore_nodes = list({node.text: node for node in lore_nodes}.values())
        
        chat_nodes = chat_retriever.retrieve(ev.message)
        chat_nodes_bm25 = chat_bm25_retriever.retrieve(ev.message)
        chat_nodes.extend(chat_nodes_bm25)
        # deduplicate chat nodes
        chat_nodes = list({node.text: node for node in chat_nodes}.values())

        # Combine context from both sources
        lore_context = "\n".join([f"""[LORE] {node.metadata["headings"]}: {node.text}""" for node in lore_nodes])
        chat_context = "\n".join([f"[CHAT] {node.text}" for node in chat_nodes])

        combined_context = f"{lore_context}\n\n{chat_context}"

        prompt = f"""{SYSTEM_PROMPT.template}\n\n
                    {CHARACTER_PROMPT.template}\n\n
                    ##CONTESTO AGGIUNTIVO:
                    {combined_context}\n\n
                    ##CONVERSAZIONI PRECEDENTI (ultimi 10 messaggi):
                    {last_10_messages}\n\n
                    ##MESSAGGIO DEL GIOCATORE: 
                    {ev.message}\n
                    ##IL TUO PERSONAGGIO:"""

        print(f"\nprompt length: {len(tiktoken.encoding_for_model('gpt-4o-mini').encode(prompt))} tokens\n")
        print(f"\nGenerating response \n--------------\n")

        # se nel messaggio c'è <debug> allora stampo il prompt
        if ev.verbose:
            print(f"Prompt for LLM:\n{prompt}\n--------------\n")

        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            # Save the exchange to chat memory
            await self._save_to_chat_memory(ev.message, response_text)
            
            await self._update_running_story(ctx, f"\nIl tuo personaggio: {response_text}")
            return AssistantResponseEvent(response=response_text)
        except Exception as e:
            print(f"Error generating response: {e}")
            return StopEvent()

    # @step
    # async def should_continue(self, ctx: Context, ev: AssistantResponseEvent) -> StopEvent | AssistantResponseEvent:
    #     continue_chat = input("\nContinue chatting? (yes/no): ").lower()
    #     if continue_chat == "yes":
    #         return ev
    #     else:
    #         print("Ending the chat.")
    #         return StopEvent()