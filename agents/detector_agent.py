# -*- coding: utf-8 -*-
import os
import json
import logging
from langchain_community.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_base.bias_store import CognitiveBiasStore

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DetectorAgent:
    """
    Интеллектуальный агент для диагностики когнитивных искажений (Контур Б).
    Использует LLM (GigaChat) для семантического анализа текста на основе
    библиотеки когнитивных искажений, подгружаемой через RAG.
    """
    def __init__(self):
        """
        Инициализирует агента, модель GigaChat и хранилище когнитивных искажений.
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

        # Инициализация RAG-хранилища для искажений
        try:
            self.bias_store = CognitiveBiasStore()
            logging.info("Хранилище когнитивных искажений (CognitiveBiasStore) инициализировано.")
        except Exception as e:
            logging.error(f"Ошибка при инициализации CognitiveBiasStore: {e}")
            self.bias_store = None


    def _create_system_prompt(self, relevant_biases: list) -> str:
        """
        Создает системный промпт с динамическим списком релевантных когнитивных искажений.
        """
        if relevant_biases:
            biases_description = "\n".join([
                f"- {bias['name']}: {bias['description']}"
                for bias in relevant_biases
            ])
            bias_instruction = f"""Проанализируй текст на наличие признаков когнитивных искажений из этого списка наиболее вероятных кандидатов:
{biases_description}"""
        else:
            bias_instruction = "Проанализируй текст на наличие признаков любых известных когнитивных искажений. В твоем ответе могут быть искажения, даже если их нет в предоставленном списке."


        prompt = f"""
Ты — опытный психолог-лингвист. Твоя задача — провести глубокий психолингвистический анализ текста пользователя по трем направлениям: когнитивные искажения, эмоциональный тон и стиль коммуникации.

**1. Когнитивные искажения:**
{bias_instruction}

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
- "cognitive_biases" должен быть списком словарей (пустой список [], если искажений нет).
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

    def analyze(self, text: str) -> dict:
        """
        Анализирует текст на наличие когнитивных искажений, эмоционального тона и стиля коммуникации.
        Использует RAG для поиска релевантных искажений перед отправкой запроса LLM.
        Возвращает словарь Python с комплексным результатом.
        """
        default_response = {
            "cognitive_biases": [],
            "emotional_tone": "Нейтральный",
            "communication_style": "Аналитический"
        }

        if not self.llm or not self.bias_store:
            logging.error("LLM или CognitiveBiasStore не были инициализированы. Анализ невозможен.")
            return default_response

        if not text:
            return default_response

        # 1. RAG-поиск релевантных искажений
        relevant_biases = self.bias_store.query_biases(text, n_results=5)

        # 2. Создание динамического промпта
        system_prompt = self._create_system_prompt(relevant_biases)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Проанализируй этот текст: \"{text}\"")
        ]

        try:
            res = self.llm.invoke(messages)
            llm_response_text = res.content

            if "```json" in llm_response_text:
                llm_response_text = llm_response_text.split("```json")[1].split("```")[0].strip()

            analysis_data = json.loads(llm_response_text)

            # Новая, более надежная валидация
            if not isinstance(analysis_data, dict) or not all(k in analysis_data for k in ["cognitive_biases", "emotional_tone", "communication_style"]):
                raise ValueError("LLM-ответ имеет неверную структуру (отсутствуют ключи)")

            if not isinstance(analysis_data["cognitive_biases"], list):
                 raise ValueError("Ключ 'cognitive_biases' не является списком")


            logging.info(f"Анализ текста '{text[:50]}...' завершен.")
            return analysis_data

        except json.JSONDecodeError as e:
            logging.error(f"Ошибка декодирования JSON от LLM: {e}\nОтвет LLM:\n{llm_response_text}")
            return default_response
        except Exception as e:
            logging.error(f"Неожиданная ошибка при анализе текста в DetectorAgent: {e}")
            return default_response
