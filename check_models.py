import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

print("🔍 Interrogation des modèles accessibles sur votre compte Groq...")
try:
    models = client.models.list()
    print("✅ Modèles disponibles sur votre compte :")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"❌ Erreur : {e}")