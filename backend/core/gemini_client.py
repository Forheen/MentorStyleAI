# core/gemini_client.py

from google import genai
from backend.core.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)