# В начале файла
from sqlalchemy.orm import Session
from database.models import User, CognitivePattern, DialogueEntry, UserProfile, UserTrait
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, DefaultEmbeddingFunction
from database.db_connector import SessionLocal, chroma_client, get_chroma_collection, add_user_trait, get_user_traits
from datetime import datetime, timedelta
from sqlalchemy import desc

# Инициализация эмбеддингов
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


class DynamicMemory:
    def __init__(self, user_id_stub: str):
        self.user_id_stub = user_id_stub
        self.db_session = SessionLocal()

        # Инициализация пользователя и профиля
        self.user = self._get_or_create_user()

        # 🔥 Векторная память — история диалогов
        self.vector_collection = get_chroma_collection(f"dialogue_vector_{user_id_stub}")

        # ⚠️ history_collection — дубль? Если не используется в другом месте — можно убрать
        # self.history_collection = get_chroma_collection(f"history_{user_id_stub}")

        print(f"Пользователь {user_id_stub} инициализирован.")

    def _init_vector_collection(self):
        """Создаёт или получает коллекцию Chroma для хранения диалогов."""
        collection_name = f"dialogue_vector_{self.user_id_stub}"
        self.vector_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def _get_or_create_user(self):
        user = self.db_session.query(User).filter_by(user_id_stub=self.user_id_stub).first()
        if not user:
            user = User(user_id_stub=self.user_id_stub)
            self.db_session.add(user)
            self.db_session.commit()
        if not user.profile:
            profile = UserProfile(user_id=user.id)
            self.db_session.add(profile)
            self.db_session.commit()
            user.profile = profile
        return user

    def save_interaction(self, text: str, is_user: bool):
        """Сохраняет взаимодействие в SQLite и вектор в ChromaDB."""
        if not is_user:
            return  # Сохраняем в вектор только реплики пользователя

        try:
            # Сохранение в SQLite
            entry = DialogueEntry(user_id=self.user.id, is_user=is_user, content=text)
            self.db_session.add(entry)
            self.db_session.commit()

            # 🔥 Сохранение в ChromaDB
            self.vector_collection.add(
                ids=[str(entry.id)],
                documents=[text],
                metadatas=[{
                    "user_id": self.user.id,
                    "type": "user_input",
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else ""
                }]
            )
        except Exception as e:
            self.db_session.rollback()
            print(f"Ошибка при сохранении взаимодействия: {e}")

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
            print(f"✅ Сохранён паттерн '{pattern_name}' (уверенность: {confidence})")
        except Exception as e:
            self.db_session.rollback()
            print(f"❌ Ошибка при сохранении паттерна: {e}")

    
    def search_memories(self, query: str, n_results: int = 3) -> list:
        """Ищет похожие сообщения в памяти."""
        try:
            results = self.vector_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            # results["documents"][0] — это список релевантных текстов
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Ошибка при поиске в памяти: {e}")
            return []

    def get_last_session_summary_for_prompt(self) -> str:
        summary = self.get_last_session_summary()
        if summary:
            return f"\n\n📌 Из предыдущего разговора:\n{summary}"
        return ""

    def get_pattern_frequency(self, pattern_name: str) -> int:
        """Возвращает количество раз, сколько встречался паттерн."""
        count = self.db_session.query(CognitivePattern).filter_by(
            user_id=self.user.id,
            pattern_name=pattern_name
        ).count()
        return count

    def get_user_patterns(self):
        """Возвращает все когнитивные паттерны для текущего пользователя."""
        try:
            return self.db_session.query(CognitivePattern).filter_by(user_id=self.user.id).all()
        except Exception as e:
            print(f"Ошибка при получении паттернов: {e}")
            return []

    def get_pattern_weight(self, pattern_name: str) -> float:
        patterns = self.db_session.query(CognitivePattern).filter(...).all()
        weight = 0
        for p in patterns:
            days_ago = (datetime.utcnow() - p.observed_at).days
            decay = 0.9 ** (days_ago / 7)  # Затухание на 10% в неделю
            weight += decay
        return weight

    def get_pattern_weight_over_time(self, pattern_name: str, window_days: int = 30):
        """
        Возвращает "вес" паттерна за последние N дней с учётом затухания.
        Используется для отслеживания прогресса (ЗБР).
        """
        patterns = (
            self.db_session.query(CognitivePattern)
            .filter_by(user_id=self.user.id, pattern_name=pattern_name)
            .order_by(CognitivePattern.observed_at)
            .all()
        )

        if not patterns:
            return 0.0

        total_weight = 0.0
        now = datetime.utcnow()

        for p in patterns:
            days_ago = (now - p.observed_at).days
            if days_ago > window_days:
                continue  # вне окна
            decay = 0.9 ** (days_ago / 7)  # экспоненциальное затухание (10% в неделю)
            total_weight += decay

        return round(total_weight, 2)

    def get_pattern_history(self, pattern_name: str, limit: int = 10):
        """
        Возвращает последние N наблюдений за паттерном.
        Полезно для анализа динамики.
        """
        patterns = (
            self.db_session.query(CognitivePattern)
            .filter_by(user_id=self.user.id, pattern_name=pattern_name)
            .order_by(desc(CognitivePattern.observed_at))
            .limit(limit)
            .all()
        )
        return patterns

    def get_pattern_frequency(self, pattern_name: str) -> int:
        """
        Возвращает общее количество наблюдений за паттерном.
        Уже есть — оставляем как есть.
        """
        count = self.db_session.query(CognitivePattern).filter_by(
            user_id=self.user.id,
            pattern_name=pattern_name
        ).count()
        return count

    def get_user_profile_summary(self) -> str:
        """Возвращает краткое резюме того, что знает о пользователе."""
        summary_parts = []

        # Имя
        name = self.get_user_name()
        if name:
            summary_parts.append(f"Тебя зовут {name}.")

        # Последнее резюме
        last_summary = self.get_last_session_summary()
        if last_summary:
            summary_parts.append(f"В прошлый раз мы говорили о: {last_summary}")

        # Паттерны
        patterns = self.get_user_patterns()
        if patterns:
            unique_biases = {p.pattern_name for p in patterns}
            bias_names = {
                "black_and_white_thinking": "черно-белое мышление",
                "catastrophizing": "катастрофизация",
                "overgeneralization": "чрезмерное обобщение",
                "personalization": "персонализация"
            }
            human_biases = [bias_names.get(b, b) for b in unique_biases]
            if human_biases:
                summary_parts.append(f"Я отмечал у тебя паттерны: {', '.join(human_biases)}.")

        # Число диалогов
        recent_messages = self.db_session.query(DialogueEntry).filter_by(user_id=self.user.id).count()
        if recent_messages > 0:
            summary_parts.append(f"Мы уже обменялись {recent_messages} сообщениями.")

        # Черты пользователя
        traits_summary = self.get_user_traits_summary()
        if traits_summary:
            summary_parts.append(traits_summary)

        # Новые психолингвистические данные
        if self.user.profile and self.user.profile.last_emotional_tone:
            summary_parts.append(f"Твой последний эмоциональный тон был '{self.user.profile.last_emotional_tone}'.")
        if self.user.profile and self.user.profile.dominant_communication_style:
            summary_parts.append(f"Твой доминирующий стиль общения — '{self.user.profile.dominant_communication_style}'.")

        return " ".join(summary_parts) if summary_parts else "Пока что я мало о тебе знаю."

    def save_user_trait(self, trait_type: str, trait_description: str, confidence: int):
        """Сохраняет новую черту пользователя в БД."""
        try:
            add_user_trait(
                db_session=self.db_session,
                user_id=self.user.id,
                trait_type=trait_type,
                trait_description=trait_description,
                confidence=confidence
            )
            print(f"✅ Сохранена черта '{trait_type}': {trait_description}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении черты: {e}")

    def get_user_traits_summary(self) -> str:
        """Возвращает форматированную строку с чертами пользователя."""
        try:
            traits = get_user_traits(self.db_session, self.user.id)
            if not traits:
                return ""

            summary_parts = []
            for trait in traits:
                summary_parts.append(f"[{trait.trait_type.capitalize()}] {trait.trait_description}")

            return "Наблюдаемые черты: " + "; ".join(summary_parts) + "."
        except Exception as e:
            print(f"Ошибка при получении черт: {e}")
            return ""

    def get_full_profile_context(self) -> str:
        """
        Возвращает полный контекст о пользователе: имя, прошлые темы, паттерны, статистику.
        Используется в автонапоминании и команде /memory.
        """
        summary_parts = []

        # 1. Имя
        name = self.get_user_name()
        if name:
            summary_parts.append(f"Тебя зовут {name}.")

        # 2. Последнее резюме сессии
        last_summary = self.get_last_session_summary()
        if last_summary:
            summary_parts.append(f"В прошлый раз мы говорили о: {last_summary}")

        # 3. Когнитивные паттерны
        patterns = self.get_user_patterns()
        if patterns:
            unique_biases = {p.pattern_name for p in patterns}
            bias_names = {
                "black_and_white_thinking": "черно-белое мышление",
                "catastrophizing": "катастрофизация",
                "overgeneralization": "чрезмерное обобщение",
                "personalization": "персонализация",
                "hindsight_bias": "ошибка ретроспективного взгляда",
                "emotional_reasoning": "эмоциональное обоснование",
                "mind_reading": "чтение мыслей"
            }
            human_biases = [bias_names.get(b, b) for b in sorted(unique_biases)]
            if human_biases:
                summary_parts.append(f"Я отмечал у тебя паттерны: {', '.join(human_biases)}.")

        # 4. Активность
        message_count = self.db_session.query(DialogueEntry).filter_by(user_id=self.user.id).count()
        if message_count > 0:
            summary_parts.append(f"Мы уже обменялись {message_count} сообщениями.")

        return " ".join(summary_parts) if summary_parts else "Пока что я мало о тебе знаю."

    def save_user_name(self, name: str):
        if not self.user.profile:
            self.user.profile = UserProfile(user_id=self.user.id)
            self.db_session.add(self.user.profile)
        self.user.profile.name = name
        self.db_session.commit()

    def get_user_name(self) -> str:
        return self.user.profile.name if self.user.profile and self.user.profile.name else None

    def save_session_summary(self, summary: str):
        if not self.user.profile:
            self.user.profile = UserProfile(user_id=self.user.id)
            self.db_session.add(self.user.profile)
        self.user.profile.last_session_summary = summary
        self.db_session.commit()

    def get_last_session_summary(self) -> str:
        return self.user.profile.last_session_summary if self.user.profile and self.user.profile.last_session_summary else None

    def save_psycholinguistic_features(self, emotional_tone: str, communication_style: str):
        """
        Сохраняет последние психолингвистические метрики в профиль пользователя.
        """
        try:
            if not self.user.profile:
                # На всякий случай, если профиль еще не создан
                self.user.profile = UserProfile(user_id=self.user.id)
                self.db_session.add(self.user.profile)

            self.user.profile.last_emotional_tone = emotional_tone
            self.user.profile.dominant_communication_style = communication_style
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            print(f"❌ Ошибка при сохранении психолингвистических метрик: {e}")

    def __del__(self):
        if hasattr(self, 'db_session'):
            self.db_session.close()
