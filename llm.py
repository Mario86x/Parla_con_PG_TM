from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
import os

def init_llm(api_key):
    llm = GoogleGenAI(
        model="models/gemini-1.5-flash",
        api_key=api_key,  # uses GOOGLE_API_KEY env var by default
    )
    print("Using Google Gemini API")
    return llm

def init_local_llm():
    llm = Ollama(model="deepseek-r1:1.5b")
    return llm


if __name__ == "__main__":
    from llama_index.core.bridge.pydantic import BaseModel
    from llama_index.core.prompts import PromptTemplate

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = init_llm(api_key)
    ollama_llm = init_local_llm()
    
    class Character(BaseModel):
        """A character in a story"""

        name: str
        role: str
        motivation: str


    prompt = PromptTemplate(
        "create a character based on the following information: {text}"
    )

    # response = llm.structured_predict(
    #     Character, prompt, text="he is a knight who is on a quest to save the princess"
    # )

    # print(response)

    response_ollama = ollama_llm.structured_predict(
        Character, prompt, text="he is a knight who is on a quest to save the princess"
    )
    print(response_ollama)