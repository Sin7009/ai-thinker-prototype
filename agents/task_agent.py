import os
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

class TaskAgent:
    """
    Агент для выполнения конкретных задач (режим "Копилот").
    Использует OpenRouter.
    """
    def __init__(self, system_prompt: str = None, model_name: str = "google/gemini-2.0-flash-exp:free"):
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("Переменная окружения OPENROUTER_API_KEY не установлена.")

        # Инициализация через OpenRouter
        self.chat = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            default_headers={
                "HTTP-Referer": "https://github.com/ai-thinker",
                "X-Title": "AI Thinker Prototype"
            }
        )

        # Системный промпт по умолчанию
        self.system_prompt = system_prompt or (
            "Ты — AI-партнёр по мышлению. Твой стиль — ассертивный, эмпатичный и диагностический. "
            "Ты не просто отвечаешь на вопросы, а ведешь пользователя к решению. "
            "Бери на себя управление диалогом. Используй сильные, уверенные формулировки. "
            "Обращайся к пользователю на 'ты', если из контекста не следует иное."
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        print(f"TaskAgent инициализирован на модели: {model_name}")

    def process(self, text: str, context_memory: str = "") -> str:
        try:
            full_system_prompt = self.system_prompt
            if context_memory.strip():
                full_system_prompt += "\n\n" + context_memory.strip()

            messages = [SystemMessage(content=full_system_prompt)]
            messages.extend(self.memory.chat_memory.messages)
            messages.append(HumanMessage(content=text))

            response = self.chat.invoke(messages)

            self.memory.chat_memory.add_user_message(text)
            self.memory.chat_memory.add_ai_message(response.content)

            return response.content
        except Exception as e:
            print(f"Ошибка при обращении к LLM: {e}")
            traceback.print_exc()
            return "Извините, произошла ошибка сети или API."

    def clear_memory(self):
        self.memory.chat_memory.clear()
