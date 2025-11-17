import os
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

class TaskAgent:
    """
    Агент для выполнения конкретных задач (режим "Копилот").
    Использует GigaChat с поддержкой контекстной памяти.
    """
    def __init__(self, system_prompt: str = None):
        if 'GIGACHAT_CREDENTIALS' not in os.environ:
            raise ValueError("Переменная окружения GIGACHAT_CREDENTIALS не установлена.")

        self.chat = GigaChat(
            credentials=os.environ['GIGACHAT_CREDENTIALS'],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_PERS',
            model='GigaChat-Pro',  # Лучше качество
            temperature=0.7
        )

        # Системный промпт по умолчанию
        self.system_prompt = system_prompt or (
            "Ты — полезный AI-ассистент в режиме 'Копилот'. "
            "Твоя задача — прямо и эффективно отвечать на вопросы пользователя."
        )

        # Инициализация памяти
        self.memory = ConversationBufferMemory(return_messages=True)

        print("TaskAgent (GigaChat) инициализирован.")

    def process(self, text: str, context_memory: str = "") -> str:
        try:
            full_system_prompt = self.system_prompt
            if context_memory.strip():
                full_system_prompt += "\n\n" + context_memory.strip()

            messages = [
                SystemMessage(content=full_system_prompt),
            ]
            messages.extend(self.memory.chat_memory.messages)
            messages.append(HumanMessage(content=text))

            response = self.chat.invoke(messages)
            self.memory.chat_memory.add_user_message(text)
            self.memory.chat_memory.add_ai_message(response.content)

            return response.content
        except Exception as e:
            print(f"Ошибка при обращении к GigaChat: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    def clear_memory(self):
        """Очистка памяти"""
        self.memory.chat_memory.clear()

    def set_system_prompt(self, prompt: str):
        """Динамическое изменение поведения агента"""
        self.system_prompt = prompt