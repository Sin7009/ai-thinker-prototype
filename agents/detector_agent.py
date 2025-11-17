# -*- coding: utf-8 -*-
import os
import json
import logging
from langchain_community.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from agents.cognitive_biases import COGNITIVE_BIASES

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DetectorAgent:
    """
    Интеллектуальный агент для диагностики когнитивных искажений (Контур Б).
    Использует LLM (GigaChat) для семантического анализа текста на основе
    расширенной библиотеки когнитивных искажений.
    """
    def __init__(self):
        """
        Инициализирует агента, модель GigaChat и формирует системный промпт.
        """
        try:
            self.llm = GigaChat(
                credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS"
            )
            logging.info("DetectorAgent (LLM-based) инициализирован успешно.")
        except Exception as e:
            logging.error(f"Ошибка при инициализации GigaChat в DetectorAgent: {e}")
            self.llm = None

        self.system_prompt_template = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """
        Создает и кэширует системный промпт с полным списком когнитивных искажений.
        """
        biases_description = "\n".join([
            f"- {bias['name']}: {bias['description']}"
            for bias in COGNITIVE_BIASES
        ])

        prompt = f"""
Ты — опытный психолог-лингвист. Твоя задача — провести глубокий психолингвистический анализ текста пользователя по трем направлениям: когнитивные искажения, эмоциональный тон и стиль коммуникации.

**1. Когнитивные искажения:**
Проанализируй текст на наличие признаков когнитивных искажений из этого списка:
{biases_description}

**2. Эмоциональный тон:**
Определи доминирующую эмоцию в тексте. Выбери ОДНУ из следующих: Нейтральный, Радость, Грусть, Гнев, Страх, Удивление, Неуверенность, Раздражение.

**3. Стиль коммуникации:**
Определи основной стиль общения пользователя. Выбери ОДИН из следующих:
- **Аналитический:** Структурированный, логичный, сфокусированный на фактах и данных.
- **Эмоциональный:** Экспрессивный, сфокусированный на чувствах и личных переживаниях.
- **Директивный:** Уверенный, нацеленный на результат, дающий указания или твердо заявляющий о своей позиции.
- **Интуитивный:** Ассоциативный, образный, перескакивающий между идеями, сфокусированный на общей картине.

**Формат ответа:**
Верни результат СТРОГО в формате единого JSON-объекта.
Объект должен содержать три ключа: "cognitive_biases", "emotional_tone", "communication_style".
- "cognitive_biases" должен быть списком словарей, как в предыдущих инструкциях (пустой список [], если искажений нет).
- "emotional_tone" должен содержать одну строку с названием эмоции.
- "communication_style" должен содержать одну строку с названием стиля.

**Пример ответа:**
{{
  "cognitive_biases": [
    {{
      "name": "Катастрофизация",
      "confidence": 95,
      "context": "Пользователь предполагает наихудший исход, говоря 'если я провалю это собеседование, моя карьера закончена'."
    }}
  ],
  "emotional_tone": "Страх",
  "communication_style": "Эмоциональный"
}}

Не добавляй никаких пояснений до или после JSON-ответа. Только чистый JSON.
"""
        return prompt

    def analyze(self, text: str) -> str:
        """
        Анализирует текст на наличие когнитивных искажений, эмоционального тона и стиля коммуникации.
        Возвращает JSON-строку с комплексным результатом.
        """
        if not self.llm:
            logging.error("LLM не была инициализирована. Анализ невозможен.")
            return json.dumps({"error": "LLM not initialized"}, ensure_ascii=False, indent=2)

        if not text:
            # Возвращаем структуру по умолчанию
            return json.dumps({
                "cognitive_biases": [],
                "emotional_tone": "Нейтральный",
                "communication_style": "Аналитический"
            }, ensure_ascii=False, indent=2)

        messages = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(content=f"Проанализируй этот текст: \"{text}\"")
        ]

        try:
            res = self.llm.invoke(messages)
            llm_response_text = res.content

            # Попытка очистить ответ от возможных артефактов (например, ```json)
            if "```json" in llm_response_text:
                llm_response_text = llm_response_text.split("```json")[1].split("```")[0].strip()

            # Парсинг JSON
            detected_biases = json.loads(llm_response_text)

            # Простая валидация формата
            if not isinstance(detected_biases, list):
                raise ValueError("LLM-ответ не является списком")

            logging.info(f"Анализ текста '{text[:50]}...' завершен. Найдено искажений: {len(detected_biases)}")
            return json.dumps(detected_biases, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as e:
            logging.error(f"Ошибка декодирования JSON от LLM: {e}\nОтвет LLM:\n{llm_response_text}")
            return json.dumps({"error": "Invalid JSON response from LLM", "details": str(e)}, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Неожиданная ошибка при анализе текста в DetectorAgent: {e}")
            return json.dumps({"error": "An unexpected error occurred", "details": str(e)}, ensure_ascii=False, indent=2)

# Пример использования (для отладки)
if __name__ == '__main__':
    # Убедитесь, что GIGACHAT_CREDENTIALS установлен как переменная окружения
    if 'GIGACHAT_CREDENTIALS' not in os.environ:
        print("Ошибка: Переменная окружения GIGACHAT_CREDENTIALS не установлена.")
    else:
        agent = DetectorAgent()

        test_text_1 = "Если я не сдам этот экзамен на отлично, я полный неудачник, и вся моя учеба была зря. Это будет просто катастрофа."
        test_text_2 = "Мой друг не ответил на сообщение, он наверняка на меня обиделся."
        test_text_3 = "Сегодня прекрасный день, и я уверен, что все будет хорошо."

        print("\n--- Тест 1 ---")
        print(f"Текст: {test_text_1}")
        print("Результат анализа:")
        print(agent.analyze(test_text_1))

        print("\n--- Тест 2 ---")
        print(f"Текст: {test_text_2}")
        print("Результат анализа:")
        print(agent.analyze(test_text_2))

        print("\n--- Тест 3 ---")
        print(f"Текст: {test_text_3}")
        print("Результат анализа:")
        print(agent.analyze(test_text_3))
