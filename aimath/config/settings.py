import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # Project Paths
    BASE_DIR = Path(__file__).parent.parent
    DB_DIR = BASE_DIR / "database" / "storage"
    CHROMA_DB_DIR = DB_DIR / "chroma_new" # Changed from 'chroma' to bypass file locks
    SQLITE_DB_PATH = DB_DIR / "memory.db"
    
    # Models
    # Updated to Llama 3.3 70B Versatile as previous model was decommissioned
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile") 
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") # Local HF model by default for Chroma

    # Agent Configs
    HITL_CONFIDENCE_THRESHOLD = 0.8
    OCR_CONFIDENCE_THRESHOLD = 0.6
    
    # Keys
    # We expect GROQ_API_KEY to be in env

    @staticmethod
    def ensure_dirs():
        Settings.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        # SQLite path is a file, so ensure parent exists
        Settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create directories on import
Settings.ensure_dirs()
