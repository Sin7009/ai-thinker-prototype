import os
import traceback
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
    def __init__(self, system_prompt: str = None, model_name: str = "GigaChat-2"):
        if 'GIGACHAT_CREDENTIALS' not in os.environ:
            raise ValueError("Переменная окружения GIGACHAT_CREDENTIALS не установлена.")

        self.chat = GigaChat(
            credentials=os.environ['GIGACHAT_CREDENTIALS'],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_PERS',
            model=model_name,
            temperature=0.7
        )

        # Системный промпт по умолчанию
        self.system_prompt = system_prompt or (
            "Ты — AI-партнёр по мышлению. Твой стиль — ассертивный, эмпатичный и диагностический. "
            "Ты не просто отвечаешь на вопросы, а ведешь пользователя к решению. "
            "Твоя задача — проанализировать ситуацию, поставить 'диагноз' основной проблеме (например, 'паралич чистого листа', 'нечеткая цель', 'страх ошибки') "
            "и предложить конкретный, единственно верный следующий шаг. "
            "Бери на себя управление диалогом. Используй сильные, уверенные формулировки. "
            "Обращайся к пользователю на 'ты', если из контекста не следует иное. "
            "Твоя цель — не предоставить информацию, а спровоцировать у пользователя инсайт и действие."
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
            traceback.print_exc()
            return "Извините, произошла ошибка при обработке вашего запроса."

    def clear_memory(self):
        """Очистка памяти"""
        self.memory.chat_memory.clear()

    def set_system_prompt(self, prompt: str):
        """Динамическое изменение поведения агента"""
        self.system_prompt = prompt