from llm import LLMRouter

config = {
    "gemini": "YOUR_GEMINI_API_KEY",
    # "mistral": "YOUR_MISTRAL_API_KEY"
}

llm = LLMRouter(config)

def generate_response(query, context):
    context_text = "\n".join([c["text"] for c in context])

    prompt = f"""
    You are a financial assistant.

    Context:
    {context_text}

    Question:
    {query}

    Answer:
    """

    return llm.generate(prompt, provider="gemini")