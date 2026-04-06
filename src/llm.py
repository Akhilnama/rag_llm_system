from groq import Groq

class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GroqLLM(BaseLLM):
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class LLMRouter:
    def __init__(self, config):
        self.providers = {}

        if "groq" in config:
            self.providers["groq"] = GroqLLM(config["groq"])

    def generate(self, prompt: str, provider="groq") -> str:
        return self.providers[provider].generate(prompt)