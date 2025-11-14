from sqlalchemy.orm import Session
from database.models import User, CognitivePattern
from database.db_connector import SessionLocal

class DynamicMemory:
    """
    Управляет сохранением и извлечением данных о пользователе
    из долговременной памяти (базы данных SQLite).
    """
    def __init__(self, user_id_stub: str):
        self.user_id_stub = user_id_stub
        self.db_session: Session = SessionLocal()

        # Находим или создаем пользователя при инициализации
        self.user = self.db_session.query(User).filter_by(user_id_stub=self.user_id_stub).first()
        if not self.user:
            self.user = User(user_id_stub=self.user_id_stub)
            self.db_session.add(self.user)
            self.db_session.commit()
            print(f"Создан новый пользователь: {self.user_id_stub}")

    def save_cognitive_pattern(self, pattern_name: str, confidence: int, context: str):
        """Сохраняет обнаруженный когнитивный паттерн в базу данных."""
        try:
            new_pattern = CognitivePattern(
                user_id=self.user.id,
                pattern_name=pattern_name,
                confidence_score=confidence,
                context=context
            )
            self.db_session.add(new_pattern)
            self.db_session.commit()
            print(f"Сохранен новый паттерн '{pattern_name}' для пользователя {self.user_id_stub}.")
        except Exception as e:
            self.db_session.rollback()
            print(f"Ошибка при сохранении паттерна: {e}")

    def get_user_patterns(self):
        """Возвращает все когнитивные паттерны для текущего пользователя."""
        return self.db_session.query(CognitivePattern).filter_by(user_id=self.user.id).all()

    def __del__(self):
        """Закрывает сессию при уничтожении объекта."""
        self.db_session.close()

if __name__ == '__main__':
    # Пример использования
    memory = DynamicMemory("test_user_123")

    # Сохраняем паттерн
    memory.save_cognitive_pattern(
        pattern_name="black_and_white_thinking",
        confidence=90,
        context="Использовал слово 'всегда' в разговоре о проекте."
    )

    # Получаем и выводим все паттерны пользователя
    patterns = memory.get_user_patterns()
    if patterns:
        print(f"\nНайденные паттерны для {memory.user_id_stub}:")
        for p in patterns:
            print(f"- {p.pattern_name} (Уверенность: {p.confidence_score}%)")
    else:
        print("Паттерны не найдены.")
