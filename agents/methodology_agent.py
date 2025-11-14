import os
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

class MethodologyAgent:
    """
    Агент для выполнения сложных методологических задач (режим "Партнер").
    Использует GigaChat с продвинутыми техниками рассуждений.
    """
    def __init__(self):
        if 'GIGACHAT_CREDENTIALS' not in os.environ:
            raise ValueError("Переменная окружения GIGACHAT_CREDENTIALS не установлена.")

        self.chat = GigaChat(
            credentials=os.environ['GIGACHAT_CREDENTIALS'],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_CORP'
        )
        print("MethodologyAgent (GigaChat) инициализирован.")

    def execute(self, system_prompt: str, user_prompt: str) -> str:
        """
        Выполняет сложный запрос с заданным системным промптом.
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self.chat(messages)
            return response.content
        except Exception as e:
            print(f"Ошибка при обращении к GigaChat в MethodologyAgent: {e}")
            return "Извините, произошла ошибка при методологической обработке."

if __name__ == '__main__':
    # Пример использования
    if 'GIGACHAT_CREDENTIALS' in os.environ:
        methodology_agent = MethodologyAgent()

        deconstruction_system_prompt = (
            "Ты — AI-методолог. Твоя задача — помочь пользователю деконструировать его проблему. "
            "Преобразуй его нарратив в структурированную 'карту фактов'. "
            "Отделяй объективные факты от субъективных оценок. "
            "Представь результат в виде списка ключевых фактов."
        )

        user_narrative = "У нас все плохо с проектом. Сроки горят, команда демотивирована, и кажется, что клиент недоволен."

        result = methodology_agent.execute(deconstruction_system_prompt, user_narrative)

        print(f"Исходный нарратив: {user_narrative}")
        print("\n--- Результат деконструкции ---")
        print(result)
    else:
        print("Не удалось найти GIGACHAT_CREDENTIALS. Пропустите этот пример.")
