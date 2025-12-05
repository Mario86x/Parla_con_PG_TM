import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import logging
import asyncio
from typing import Dict, Any, Optional

# Importa le classi/funzioni dal tuo progetto, incluso il nuovo workflow
from llm_client import init_clients # non è più necessario init_llm o init_local_embed_model
from workflow import ChatWorkflow, UserMessageEvent, AssistantResponseEvent, StopEvent
from llama_index.core.workflow import Context # Necessario per l'esecuzione del workflow

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("DiscordBot")

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable not set")

# ⚠️ Nota: la chiave API di Google dovrebbe essere nel tuo .env (es. GOOGLE_API_KEY)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set or is empty")

# Inizializzazione del bot Discord
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Variabili Globali
# Usiamo un dizionario per memorizzare l'istanza del workflow, una per utente (se necessario per isolamento)
# Per questo esempio, useremo una singola istanza globale per semplicità.
global_workflow: Optional[ChatWorkflow] = None
# user_workflows: Dict[int, ChatWorkflow] = {} # Mantenere un workflow per utente è più complesso, usiamo il globale per ora.

@bot.event
async def on_ready():
    logger.info(f'{bot.user.name} si è connesso a Discord!')
    
    # Inizializza il workflow all'avvio del bot
    global global_workflow
    try:
        # ChatWorkflow gestisce internamente LLM, Embeddings, LoreStore e ChatStore
        global_workflow = ChatWorkflow(api_key=GOOGLE_API_KEY)
        logger.info("✅ ChatWorkflow inizializzato con successo.")
    except Exception as e:
        logger.error(f"❌ Errore durante l'inizializzazione del ChatWorkflow: {e}")
        # Se fallisce, il bot non sarà in grado di rispondere
        global_workflow = None

@bot.event
async def on_message(message):
    global global_workflow

    # Ignora i messaggi del bot stesso
    if message.author == bot.user:
        return

    # Controlla se il workflow è stato inizializzato
    if global_workflow is None:
        await message.channel.send("Mi dispiace, il sistema non è stato inizializzato correttamente (errore API/Redis). Controlla i log.")
        return

    user_message = message.content.strip()
    
    # Evita di processare messaggi vuoti o troppo brevi
    if not user_message:
        return

    # 1. Creazione dell'evento utente
    user_event = UserMessageEvent(message=user_message)

    try:
        # 2. Esecuzione del passo generate_response del workflow
        # Questo è il cuore del processo: ricerca ibrida, prompt building e LLM call.
        logger.info(f"Processing message from {message.author.name}: {user_message[:50]}...")
        
        # Creiamo un contesto con l'istanza del workflow
        workflow_context = Context(global_workflow) 
        
        # Chiamata al passo generate_response
        response_event = await global_workflow.generate_response(workflow_context, user_event)
        
        # 3. Gestione della risposta
        if isinstance(response_event, AssistantResponseEvent):
            response_text = response_event.response.strip()
            
            # 4. Invia la risposta su Discord
            await message.channel.send(response_text)
            
        elif isinstance(response_event, StopEvent):
            logger.info("Workflow fermato da un evento StopEvent.")
            # Puoi decidere di inviare un messaggio di chiusura qui se necessario
            
        else:
            logger.warning(f"Tipo di evento non gestito: {type(response_event)}")
            await message.channel.send("Mi dispiace, ho ricevuto un tipo di risposta inaspettato dal sistema.")

    except Exception as e:
        logger.error(f"❌ Errore durante l'esecuzione del workflow: {e}")
        await message.channel.send(
            "Mi dispiace, sto avendo qualche problema tecnico. Riprova più tardi."
        )

# Questa è la funzione principale per avviare il bot
def main():
    logger.info("Avvio del bot in corso...")
    # La funzione run blocca l'esecuzione
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    # Assicurati di avere le variabili d'ambiente nel tuo file .env 
    # (DISCORD_TOKEN e GOOGLE_API_KEY o altre chiavi LLM)
    main()