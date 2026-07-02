import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "corpus"

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 120
TOP_K = 4
