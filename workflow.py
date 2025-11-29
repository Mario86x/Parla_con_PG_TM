from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llama_index.core import SimpleDirectoryReader, Settings
import logging
from typing import Any
from datetime import datetime

from llm_client import init_clients, embed_text, generate_text
from prompt_builder import build_prompt
from vector_stores.lore_store import LoreStore
from vector_stores.chat_store import ChatStore
from retriever import HybridRetriever

# Events
class UserMessageEvent(Event):
    message: str

class AssistantResponseEvent(Event):
    response: str

logging.basicConfig(level=logging.DEBUG)

class ChatWorkflow(Workflow):
    def __init__(self, api_key: str):
        super().__init__()
        self.llm, self.embed_model = init_clients(api_key)
        self.lore = LoreStore()
        self.chat = ChatStore()
        self.retriever = HybridRetriever(self.lore, self.chat)
        self.running_story = ""  # Store conversation history in instance variable
        logging.info("Chat workflow initialized with hybrid retriever")

    @step
    async def process_user_input(self, ctx: Context, ev: StartEvent) -> UserMessageEvent:
        """Handle user input and create UserMessageEvent"""
        try:
            user_message = input("\n> You: ").strip()
            if not user_message:
                return StopEvent()
            return UserMessageEvent(message=user_message)
        except EOFError:
            return StopEvent()

    @step
    async def generate_response(self, ctx: Context, ev: UserMessageEvent) -> AssistantResponseEvent | StopEvent:
        """Main response generation step"""
        try:
            logging.debug("generate_response START")
            # Use instance variable for running story
            running_story = self.running_story
            logging.debug("running_story length=%d", len(running_story))

            # Generate embedding for user message
            logging.debug("Requesting embedding for message: %s", ev.message[:120])
            embedding = embed_text(self.embed_model, ev.message)
            logging.debug("Embedding received: %s", "YES" if embedding else "NO")
            if not embedding:
                logging.error("Failed to generate embedding for user message")
                return AssistantResponseEvent(response="Mi dispiace, ho avuto un errore tecnico.")
            
            # Retrieve from both stores using hybrid approach
            logging.debug("Retrieving lore (hybrid)...")
            lore_nodes = self.retriever.retrieve_lore_hybrid(
                query=ev.message,
                embedding=embedding,
                top_k=5,
                alpha=0.6
            )
            logging.debug("lore_nodes count=%d", len(lore_nodes))
            logging.debug("Retrieving chat nodes (vector)...")
            chat_nodes = self.retriever.retrieve_chat_vector(embedding, top_k=3)
            logging.debug("chat_nodes count=%d", len(chat_nodes))
            
            # Build prompt with hybrid results
            logging.debug("Building prompt...")
            prompt = build_prompt(running_story, lore_nodes, chat_nodes, ev.message)
            logging.debug("Prompt length=%d", len(prompt))
            
            # Generate response
            logging.debug("Calling LLM...")
            response_text = generate_text(self.llm, prompt)
            logging.debug("LLM responded: %s", (response_text[:120] if response_text else "<empty>"))
            if not response_text:
                logging.error("LLM generated empty response")
                return AssistantResponseEvent(response="Mi dispiace, non ho una risposta al momento.")
            
            # Save to chat memory
            logging.debug("Preparing to upsert chat memory...")
            exchange_text = f"User: {ev.message}\nAssistant: {response_text}"
            exchange_id = self.chat.create_id(ev.message, response_text)
            metadata = self.chat.create_metadata(ev.message, response_text)
            exchange_embedding = embed_text(self.embed_model, exchange_text)
            logging.debug("Exchange embedding available: %s", "YES" if exchange_embedding else "NO")
            
            if exchange_embedding:
                ok = self.chat.upsert(exchange_id, exchange_text, metadata, exchange_embedding)
                logging.debug("Upsert result: %s", ok)
            else:
                logging.warning("Skipping upsert: no embedding")
            
            # Update running story
            self.running_story += f"\n\nUser: {ev.message}\nCharacter: {response_text}"
            logging.debug("Updated running_story length=%d", len(self.running_story))
            
            logging.info("Generated response (vector+BM25): %s", response_text[:100])
            return AssistantResponseEvent(response=response_text)
        
        except Exception as e:
            logging.exception("Error in generate_response")
            return AssistantResponseEvent(response="Mi dispiace, ho avuto un errore inaspettato.")

    @step
    async def display_response(self, ctx: Context, ev: AssistantResponseEvent) -> UserMessageEvent | StopEvent:
        """Display response and ask for next input"""
        try:
            print(f"\n> Character: {ev.response}")
            user_message = input("\n> You: ").strip()
            if not user_message:
                return StopEvent()
            return UserMessageEvent(message=user_message)
        except EOFError:
            return StopEvent()