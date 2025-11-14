import os
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

class TaskAgent:
    """
    Агент для выполнения конкретных задач (режим "Копилот").
    Использует GigaChat для генерации ответов.
    """
    def __init__(self):
        # Проверяем, что токен GigaChat доступен
        if 'GIGACHAT_CREDENTIALS' not in os.environ:
            raise ValueError("Переменная окружения GIGACHAT_CREDENTIALS не установлена.")

        self.chat = GigaChat(
            credentials=os.environ['GIGACHAT_CREDENTIALS'],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_CORP'
        )
        print("TaskAgent (GigaChat) инициализирован.")

    def process(self, text: str) -> str:
        """
        Обрабатывает запрос пользователя и возвращает ответ от GigaChat.
        """
        try:
            messages = [
                SystemMessage(
                    content="Ты — полезный AI-ассистент в режиме 'Копилот'. Твоя задача — прямо и эффективно отвечать на вопросы пользователя."
                ),
                HumanMessage(content=text),
            ]
            response = self.chat(messages)
            return response.content
        except Exception as e:
            print(f"Ошибка при обращении к GigaChat: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

if __name__ == '__main__':
    # Пример использования
    # Для локального запуска установите GIGACHAT_CREDENTIALS
    if 'GIGACHAT_CREDENTIALS' in os.environ:
        task_agent = TaskAgent()
        prompt = "Что такое теория относительности?"
        answer = task_agent.process(prompt)
        print(f"Запрос: {prompt}")
        print(f"Ответ: {answer}")
    else:
        print("Не удалось найти GIGACHAT_CREDENTIALS. Пропустите этот пример.")
