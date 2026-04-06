import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# Load .env correctly
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # ✅ CURRENT MODEL
    messages=[
        {"role": "user", "content": "Explain credit risk simply"}
    ]
)

print(response.choices[0].message.content)