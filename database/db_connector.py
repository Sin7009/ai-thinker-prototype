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

# --- Функции для работы с UserTrait ---
from .models import User, UserTrait

def add_user_trait(db_session, user_id: int, trait_type: str, trait_description: str, confidence: int):
    """Добавляет новую черту пользователя."""
    new_trait = UserTrait(
        user_id=user_id,
        trait_type=trait_type,
        trait_description=trait_description,
        confidence=confidence
    )
    db_session.add(new_trait)
    db_session.commit()
    return new_trait

def get_user_traits(db_session, user_id: int) -> list[UserTrait]:
    """Возвращает все черты для указанного пользователя."""
    return db_session.query(UserTrait).filter(UserTrait.user_id == user_id).all()


print("Инициализация подключений к базам данных завершена.")
