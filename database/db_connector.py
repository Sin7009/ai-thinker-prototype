import sqlalchemy
import chromadb
from sqlalchemy.orm import sessionmaker
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import Optional

# --- SQLite (замена для PostgreSQL) ---
DB_FILE = "agent_memory.db"
engine = sqlalchemy.create_engine(f"sqlite:///{DB_FILE}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """Возвращает сессию для работы с базой данных SQLite."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ChromaDB (локальная) ---
CHROMA_PATH = "chroma_storage"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
default_embedding = DefaultEmbeddingFunction()

# Пример коллекции
# В реальном приложении коллекции будут создаваться и управляться динамически
try:
    dialogue_collection = chroma_client.get_or_create_collection("dialogue_history")
except Exception as e:
    print(f"Ошибка при создании/получении коллекции ChromaDB: {e}")

# В файле, например, `database/db_connector.py`


def get_chroma_collection(collection_name: str):
    """
    Возвращает существующую или создаёт новую коллекцию в ChromaDB.
    """
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_embedding
    )


print("Инициализация подключений к базам данных завершена.")
