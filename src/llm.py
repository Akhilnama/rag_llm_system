class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
    
# 🚀 Step 2 — Gemini Implementation (Primary)
import google.generativeai as genai

class GeminiLLM(BaseLLM):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
# 🚀 Step 3 — Mistral Implementation (Optional but good)
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class MistralLLM(BaseLLM):
    def __init__(self, api_key):
        self.client = MistralClient(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model="mistral-small",
            messages=[ChatMessage(role="user", content=prompt)]
        )
        return response.choices[0].message.content

# 🚀 Step 4 — Provider Router (CORE)
class LLMRouter:
    def __init__(self, config):
        self.providers = {}

        if "gemini" in config:
            self.providers["gemini"] = GeminiLLM(config["gemini"])

        if "mistral" in config:
            self.providers["mistral"] = MistralLLM(config["mistral"])

    def generate(self, prompt: str, provider="gemini") -> str:
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured")

        return self.providers[provider].generate(prompt)