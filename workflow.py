from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llm import init_llm
from templates import SYSTEM_PROMPT, CHARACTER_PROMPT
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
import os

PERSIST_DIR = "storage"

class UserMessageEvent(Event):
    message: str

class AssistantResponseEvent(Event):
    response: str

class ChatWorkflow(Workflow):
    def __init__(self, api_key: str):
        super().__init__()
        self.llm = init_llm(api_key)
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text:v1.5")
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        print("Using Ollama embedding model for vector store.")
        self.vector_store_index = self._load_vector_store()  # Load the vector store during initialization

        print("Chat workflow initialized with LLM and vector store.")

    def _load_vector_store(self) -> VectorStoreIndex:
        """Loads the vector store from disk or creates it if it doesn't exist."""
        # Load the existing index from disk
        print("Loading existing vector store index from disk")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Existing vector store index loaded successfully.")
        return index

    async def _update_running_story(self, ctx: Context, new_content: str):
        running_story = await ctx.get("running_story", "")
        running_story += f"\n\n{new_content}"
        await ctx.set("running_story", running_story)
    
    @step
    async def start_chat(self, ctx: Context, ev: StartEvent) -> UserMessageEvent:
        print("Welcome to the chat! Let's create a story together.")
        await self._update_running_story(ctx, "Story started.")
        print("Using Ollama embedding model for vector store.")
        return UserMessageEvent(message="Start chat")

    @step
    async def get_user_message(self, ctx: Context, ev: AssistantResponseEvent) -> UserMessageEvent:
        print("\nPrevious response:")
        print(ev.response)

        user_message = input("\nYour message: ")
        await self._update_running_story(ctx, f"\nUser: {user_message}")
        return UserMessageEvent(message=user_message)

    @step
    async def generate_response(self, ctx: Context, ev: UserMessageEvent) -> AssistantResponseEvent | StopEvent:
        running_story = await ctx.get("running_story", "")

        # Query the vector store for relevant information
        retriever = self.vector_store_index.as_retriever(similarity_top_k=3) # top k da scegliere in futuro
        nodes = retriever.retrieve(ev.message)
        context = "\n".join([node.text for node in nodes])

        prompt = f"""{SYSTEM_PROMPT.template}\n\n
                    {CHARACTER_PROMPT.template}\n\n
                    Conversazioni precedenti: {running_story}\n\n
                    Informazioni di Contesto: {context}\n\n
                    Messaggio del giocatore: {ev.message}\n
                    Il tuo personaggio:"""

        print(f"\nGenerating response \n--------------\n")

        # se nel messaggio c'è <debug> allora stampo il prompt
        if "<debug>" in ev.message:
            print(f"Prompt for LLM:\n{prompt}\n--------------\n")

        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
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