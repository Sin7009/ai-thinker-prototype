import os
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from database.db_connector import chroma_client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Инициализация эмбеддингов
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

class MethodologyAgent:
    """
    Агент для выполнения сложных методологических задач (режим "Партнер").
    Использует GigaChat с продвинутыми техниками рассуждений и контекстной памятью.
    """
    def __init__(self, user_id: str = "default_user"):
        if 'GIGACHAT_CREDENTIALS' not in os.environ:
            raise ValueError("Переменная окружения GIGACHAT_CREDENTIALS не установлена.")

        self.chat = GigaChat(
            credentials=os.environ['GIGACHAT_CREDENTIALS'],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_PERS',
            model='GigaChat-Pro',
            temperature=0.7
        )

        # 🔥 Векторная память: ChromaDB
        collection_name = f"methodology_memory_{user_id}"
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        # Храним сообщения локально для контекста
        self.message_history = []  # [(role, content), ...]

        print("MethodologyAgent (GigaChat + ChromaDB) инициализирован.")

    def execute(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # 🔍 Поиск похожих сообщений в памяти
            results = self.collection.query(
                query_texts=[user_prompt],
                n_results=3
            )
            relevant_contexts = results["documents"][0] if results["documents"] else []

            # Формируем полный промпт
            context_block = ""
            if relevant_contexts:
                context_block = "📌 Из предыдущих разговоров:\n" + "\n".join([
                    f"- {ctx}" for ctx in relevant_contexts
                ])

            full_system_prompt = system_prompt
            if context_block:
                full_system_prompt += "\n\n" + context_block

            # Добавляем краткую историю последних сообщений
            recent_context = "\n".join([
                f"{role}: {content}" for role, content in self.message_history[-4:]
            ])

            messages = [
                SystemMessage(content=full_system_prompt),
            ]

            if recent_context.strip():
                messages.append(HumanMessage(content=f"Предыдущий контекст:\n{recent_context}"))

            messages.append(HumanMessage(content=user_prompt))

            # Отправляем в модель
            response = self.chat.invoke(messages)

            # 🔥 Сохраняем в векторную память
            self.collection.add(
                ids=[f"user_{len(self.message_history)}"],
                documents=[user_prompt],
                metadatas=[{"role": "user", "type": "input"}]
            )
            self.collection.add(
                ids=[f"ai_{len(self.message_history)}"],
                documents=[response.content],
                metadatas=[{"role": "ai", "type": "response"}]
            )

            # Обновляем локальную историю
            self.message_history.append(("user", user_prompt))
            self.message_history.append(("ai", response.content))

            return response.content

        except Exception as e:
            print(f"Ошибка при обращении к GigaChat в MethodologyAgent: {e}")
            return "Извините, произошла ошибка при методологической обработке."
