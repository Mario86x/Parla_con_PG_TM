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

logging.basicConfig(level=logging.INFO)

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
                top_k=10,
                alpha=0.6
            )
            logging.debug("lore_nodes count=%d", len(lore_nodes))
            
            # MODIFICA: Utilizzo della ricerca IBRIDA per la chat history
            logging.debug("Retrieving chat nodes (hybrid)...")
            chat_nodes = self.retriever.retrieve_chat_hybrid(
                query=ev.message,
                embedding=embedding,
                top_k=5, # Meno nodi chat rispetto alla Lore per mantenere il contesto snello
                alpha=0.4 # Un alpha leggermente più basso per chat per bilanciare semantica e keyword
            )
            logging.debug("chat_nodes count=%d", len(chat_nodes))
            
            # Build prompt with hybrid results
            logging.debug("Building prompt...")
            prompt = build_prompt(running_story, lore_nodes, chat_nodes, ev.message)
            logging.debug(f"Prompt built:\n{prompt[:500]}...")  # Log first 500 chars of prompt

            # -----------------------------------------------------------------
            # LOGICA MANCANTE: Chiamata LLM, Upsert e Return
            # -----------------------------------------------------------------
            
            # Generate response from LLM
            logging.info("Calling LLM...")
            response_text = generate_text(self.llm, prompt)
            
            # Save the exchange to chat store for memory retrieval
            exchange_text = f"User: {ev.message}\nCharacter: {response_text}"
            exchange_id = self.chat.create_id(ev.message, response_text)
            metadata = self.chat.create_metadata()
            
            exchange_embedding = embed_text(self.embed_model, exchange_text)
            logging.debug("Exchange embedding available: %s", "YES" if exchange_embedding else "NO")
            
            if exchange_embedding:
                # Assumendo che il tuo ChatStore abbia un metodo upsert
                ok = self.chat.upsert(exchange_id, exchange_text, metadata, exchange_embedding)
                logging.debug("Upsert result: %s", ok)
            else:
                logging.warning("Skipping upsert: no embedding")
            
            # Update running story
            self.running_story += f"\n\nUser: {ev.message}\nCharacter: {response_text}"
            logging.debug("Updated running_story length=%d", len(self.running_story))
            
            logging.info("Generated response: %s...", response_text[:100].strip())
            return AssistantResponseEvent(response=response_text)
        
        # -----------------------------------------------------------------
        # BLOCCO EXCEPT MANCANTE: Gestione dell'errore (Indentation: 8 spazi)
        # -----------------------------------------------------------------
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
        


### test del workflow

import asyncio
import logging
from llama_index.core.workflow import Context, StopEvent # Import richiesto per il contesto nel test
import os
from dotenv import load_dotenv

# --- Funzione Asincrona per il Test ---

async def run_test_workflow():
    """
    Funzione per testare l'inizializzazione del ChatWorkflow 
    e l'esecuzione del passo generate_response.
    """
    # Usa una API key fittizia, ma DEVI assicurarti che Ollama/LLM e Redis siano attivi
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    logging.info("--- Inizializzazione Workflow ---")
    try:
        # La classe ChatWorkflow proverà a connettersi ai Redis Store all'inizializzazione
        workflow = ChatWorkflow(api_key=API_KEY)
    except Exception as e:
        logging.error(f"❌ ERRORE CRITICO: Impossibile inizializzare il Workflow.")
        logging.error(f"Dettagli: {e}")
        logging.error("Assicurati che Redis Stack, Ollama e i dati negli store siano pronti.")
        return

    # 1. Simula la query di test
    test_query = "Quali minerali preziosi sono stati menzionati nella nostra ultima conversazione?"
    
    # 2. Crea l'evento utente
    user_event = UserMessageEvent(message=test_query)
    logging.info(f"Query di test: '{test_query}'")
    
    # 3. Esegui il passo generate_response (dove avviene la ricerca ibrida)
    logging.info("--- Esecuzione generate_response (Test Ricerca Ibrida) ---")
    
    # Esegui il passo asincrono
    # CORREZIONE: Passare l'istanza 'workflow' al costruttore Context
    response_event = await workflow.generate_response(Context(workflow), user_event) 
    
    # 4. Verifica il risultato
    if isinstance(response_event, AssistantResponseEvent):
        logging.info("--- RISULTATO DEL WORKFLOW ---")
        logging.info("Risposta generata dal modello (simulata):")
        # Stampa i primi 200 caratteri della risposta
        print(f"\n> Assistant: {response_event.response.strip()}...")
        print("\n✅ Test del flusso di generazione completato con successo (LLM risponde).")
        
    elif isinstance(response_event, StopEvent):
        logging.info("❌ Il workflow si è fermato (Verifica i log per un errore, es. embedding fallito).")
    else:
        logging.error("❌ Risposta del workflow non attesa.")

# --- Blocco di Esecuzione Principale ---

if __name__ == "__main__":
    # Imposta il livello di logging su INFO per vedere il flusso e i risultati
    logging.basicConfig(level=logging.INFO) 
    
    # Esegui la funzione di test asincrona
    asyncio.run(run_test_workflow())