from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
import os

def init_llm(api_key):
    try:
        llm = GoogleGenAI(
            model="models/gemini-1.5-flash",
            api_key=api_key,  # uses GOOGLE_API_KEY env var by default
        )
        # Test the API to ensure it's working
        try:
            llm.complete("test")
        except Exception as e:
            if "billing" in str(e).lower() or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                print(f"Gemini API call succeeded initially, but failed due to billing/quota issues: {e}")
                print("Falling back to Ollama local LLM")
                return Ollama(model="mistral")
            else:
                raise e  # Re-raise the exception if it's not billing/quota related
        print("Using Google Gemini API")
        return llm
    except Exception as e:
        print(f"Failed to initialize Google Gemini API: {e}")
        print("Falling back to Ollama local LLM")
        llm = Ollama(model="deepseek-r1:1.5b")
        return llm

if __name__ == "__main__":
    from typing import List, Optional
    from llama_index.core.bridge.pydantic import BaseModel, Field
    from llama_index.core.prompts import PromptTemplate

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = init_llm(api_key)
    
    class Character(BaseModel):
        """A character in a story"""

        name: str
        role: str
        motivation: str


    prompt = PromptTemplate(
        "create a character based on the following information: {text}"
    )

    response = llm.structured_predict(
        Character, prompt, text="he is a knight who is on a quest to save the princess"
    )

    print(response)