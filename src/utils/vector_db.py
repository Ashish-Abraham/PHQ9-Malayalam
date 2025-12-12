import os
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def get_qdrant_client():
    """Returns an instance of QdrantClient configured for in-memory usage."""
    return QdrantClient(path=str(Path.home() / "phq9"))
