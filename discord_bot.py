import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from llm import init_llm, init_embed_model
from llama_index.core import StorageContext, load_index_from_storage, Settings
from templates import SYSTEM_PROMPT, CHARACTER_PROMPT
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable not set")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and Vector Store
PERSIST_DIR = "storage"
LOGS_FILE = "discord_chat_logs.json"
user_conversations = {}

def load_vector_store():
    """Load the existing vector store from disk"""
    if not os.path.exists(PERSIST_DIR):
        raise ValueError(f"Storage directory '{PERSIST_DIR}' not found")
    
    print("Loading vector store...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Vector store loaded successfully")
    return index

def save_to_logs(user_id: int, prompt: str, response_text: str):
    """Save chat interaction to JSON log file"""
    try:
        # Load existing logs
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = {}

        # Convert user_id to string for JSON compatibility
        user_id = str(user_id)
        
        if user_id not in logs:
            logs[user_id] = []
        
        logs[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response_text
        })
        
        with open(LOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logging.error(f"Error saving to logs: {e}")

# Initialize resources
llm = init_llm(GOOGLE_API_KEY)
embed_model = init_embed_model(GOOGLE_API_KEY)
Settings.llm = llm
Settings.embed_model = embed_model
vector_store = load_vector_store()

# Create bot instance
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='start')
async def start(ctx):
    """Handle the !start command"""
    user_id = ctx.author.id
    user_conversations[user_id] = []
    await ctx.send("Benvenuto! Sono il tuo personaggio RPG. Puoi iniziare a chattare con me ora.")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands (like !start)
    await bot.process_commands(message)

    # Only respond to non-command messages in DMs or when mentioned
    if not message.content.startswith('!') and (
        isinstance(message.channel, discord.DMChannel) or 
        bot.user in message.mentions
    ):
        user_id = message.author.id
        if user_id not in user_conversations:
            user_conversations[user_id] = []
        
        # Remove bot mention from message
        user_message = message.content.replace(f'<@{bot.user.id}>', '').strip()
        
        try:
            # Query vector store for context
            retriever = vector_store.as_retriever(similarity_top_k=10)
            nodes = retriever.retrieve(user_message)
            relevant_context = "\n".join([node.text for node in nodes])
            
            # Build conversation history
            conversation_history = "\n".join(user_conversations[user_id][-5:])
            
            # Create prompt
            prompt = f"""
            {SYSTEM_PROMPT.template}\n
            {CHARACTER_PROMPT.template}\n
            Informazioni di Contesto: {relevant_context}\n
            Cronologia della conversazione: {conversation_history}\n
            Messaggio del giocatore: {user_message}\n
            Il tuo personaggio:\n"""
            
            # Generate response
            response = llm.complete(prompt)
            response_text = response.text.strip()
            
            # Update conversation history
            user_conversations[user_id].append(f"Giocatore: {user_message}")
            user_conversations[user_id].append(f"Personaggio: {response_text}")
            
            # Log the interaction
            save_to_logs(user_id, prompt, response_text)
            
            # Send response
            await message.channel.send(response_text)
            
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            await message.channel.send(
                "Mi dispiace, sto avendo qualche problema. Riprova più tardi."
            )

def main():
    """Start the bot"""
    print("Starting bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()