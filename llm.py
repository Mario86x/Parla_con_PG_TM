from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
import os

def init_llm(api_key):
    llm = GoogleGenAI(
        # model="models/gemini-1.5-flash",
        # model = "models/gemma-3-27b-it",
        model = "models/gemini-2.5-flash",
        api_key=api_key,  # uses GOOGLE_API_KEY env var by default
    )
    print("Using Google Gemini API")
    return llm

def init_local_llm():
    llm = Ollama(model="deepseek-r1:1.5b")
    return llm

def init_local_embed_model(api_key):
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        api_key=api_key,  # uses GOOGLE_API_KEY env var by default
        embed_batch_size=500
    )
    print("Using Google Embedding API")
    return embed_model

def init_local_embed_model():
    embed_model = OllamaEmbedding(model_name="embeddinggemma")
    return embed_model

if __name__ == "__main__":
    from llama_index.core.bridge.pydantic import BaseModel
    from llama_index.core.prompts import PromptTemplate
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"API Key: {api_key}")
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if api_key is None:
        raise ValueError("No API key provided and GOOGLE_API_KEY environment variable not set")

    # llm = init_llm(api_key)
    # ollama_llm = init_local_llm()
    
    # class Character(BaseModel):
    #     """A character in a story"""

    #     name: str
    #     role: str
    #     motivation: str


    # prompt = PromptTemplate(
    #     "create a character based on the following information: {text}"
    # )

    # response = llm.structured_predict(
    #     Character, prompt, text="he is a knight who is on a quest to save the princess"
    # )

    # print(response)

    # response_ollama = ollama_llm.structured_predict(
    #     Character, prompt, text="he is a knight who is on a quest to save the princess"
    # )
    # print(response_ollama)

    embed_model = init_local_embed_model(api_key)
    embeddings = embed_model.get_text_embedding("Mario")
    print(embeddings[:5])
    print(f"Dimension of embeddings: {len(embeddings)}")