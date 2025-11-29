from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
from llm import init_llm, init_local_embed_model
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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
print(TELEGRAM_TOKEN)
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable not set")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and Vector Store
PERSIST_DIR = "storage"
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

# Initialize resources
llm = init_llm(GOOGLE_API_KEY)
embed_model = init_local_embed_model(GOOGLE_API_KEY)
Settings.llm = llm
Settings.embed_model = embed_model
vector_store = load_vector_store()
print("loading complete")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    user_id = update.effective_user.id
    user_conversations[user_id] = []
    
    await update.message.reply_text(
        "Benvenuto! Sono il tuo personaggio RPG. Puoi iniziare a chattare con me ora."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages"""
    user_id = update.effective_user.id
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    
    # Get user's message
    user_message = update.message.text
    
    try:
        # Query vector store for context
        retriever = vector_store.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(user_message)
        relevant_context = "\n".join([node.text for node in nodes])
        
        # Build conversation history
        conversation_history = "\n".join(user_conversations[user_id][-5:])  # Last 5 messages
        
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
        await update.message.reply_text(response_text)
        
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "Mi dispiace, sto avendo qualche problema. Riprova più tardi."
        )



# Add this after user_conversations initialization
LOGS_FILE = "chat_logs.json"

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
        
        # Initialize user list if not exists
        if user_id not in logs:
            logs[user_id] = []
        
        # Add new interaction
        logs[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response_text
        })
        
        # Save updated logs
        with open(LOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logging.error(f"Error saving to logs: {e}")

def main():
    """Start the bot"""
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()