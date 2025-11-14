from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    user_id_stub = Column(String, unique=True, index=True, default="default_user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cognitive_patterns = relationship("CognitivePattern", back_populates="user")

class CognitivePattern(Base):
    __tablename__ = 'cognitive_patterns'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    pattern_name = Column(String, index=True)
    context = Column(Text)
    confidence_score = Column(Integer)
    observed_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="cognitive_patterns")

# --- Создание таблиц ---
# Этот код будет выполнен при первом импорте, создавая таблицы, если их нет
from .db_connector import engine
Base.metadata.create_all(bind=engine)

print("Модели SQLAlchemy и таблицы созданы.")
