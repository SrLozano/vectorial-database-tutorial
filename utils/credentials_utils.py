"""
Configuration utilities for local RAG setup.
Simplified to support only local Milvus Lite and sentence-transformers.
"""
import logging
from os import environ

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def get_config_value(key: str, default: str = None) -> str:
    """Get configuration value from environment variables.

    This function reads directly from os.environ to ensure
    we always get the latest values, even if load_dotenv()
    was called after module import.
    """
    return environ.get(key, default)


class LocalConfig:
    """Configuration for local RAG components."""
    
    def __init__(self):
        # Milvus Lite database path (local file)
        self.milvus_db_path = get_config_value("MILVUS_DB_PATH", "./milvus_demo.db")
        
        # Sentence-transformers model name
        self.embedding_model = get_config_value(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
