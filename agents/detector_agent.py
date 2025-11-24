import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_base.bias_store import CognitiveBiasStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DetectorAgent:
    """
    Агент для диагностики (Контур Б).
    Использует дешевую/быструю модель через OpenRouter.
    """
    def __init__(self, model_name: str = "google/gemini-2.0-flash-exp:free"):
        try:
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get('OPENROUTER_API_KEY'),
                model=model_name,
                temperature=0.1, # Низкая температура для анализа
                default_headers={
                    "HTTP-Referer": "https://github.com/ai-thinker",
                    "X-Title": "AI Thinker Detector"
                }
            )
            self.bias_store = CognitiveBiasStore()
            logging.info(f"DetectorAgent инициализирован ({model_name}).")
        except Exception as e:
            logging.error(f"Ошибка инициализации DetectorAgent: {e}")
            self.llm = None
            self.bias_store = None

    def _verify_bias(self, text, suspected_bias):
        """Верификация гипотезы (Адвокат Дьявола)."""
        verification_prompt = (
            f"Текст: «{text}»\n"
            f"Гипотеза: Здесь есть искажение '{suspected_bias}'.\n"
            "Твоя задача — найти аргументы ПРОТИВ этой гипотезы. "
            "Если сомнения сильны, верни FALSE. Если искажение очевидно, верни TRUE."
        )
        try:
            messages = [
                SystemMessage(content="You are a skeptical psychologist. Output only TRUE or FALSE."),
                HumanMessage(content=verification_prompt)
            ]
            res = self.llm.invoke(messages)
            return "TRUE" in res.content.strip().upper()
        except Exception:
            return False

    def _create_system_prompt(self, relevant_biases: list) -> str:
        if relevant_biases:
            biases_desc = "\n".join([f"- {b['name']}: {b['description']}" for b in relevant_biases])
            instruction = f"Список кандидатов:\n{biases_desc}"
        else:
            instruction = "Ищи любые известные когнитивные искажения."

        return f"""
Ты — психолог-лингвист. Анализируй текст на: когнитивные искажения, эмоциональный тон, стиль.
1. Искажения: {instruction}
2. Тон: (Нейтральный, Радость, Гнев, Страх, Грусть).
3. Стиль: (Аналитический, Эмоциональный, Директивный).

Верни ТОЛЬКО JSON:
{{
  "cognitive_biases": [{{"name": "...", "confidence": 90, "context": "..."}}],
  "emotional_tone": "...",
  "communication_style": "..."
}}
"""

    def analyze(self, text: str) -> dict:
        default_response = {"cognitive_biases": [], "emotional_tone": "Нейтральный", "communication_style": "Аналитический"}

        if not self.llm or not text:
            return default_response

        relevant_biases = self.bias_store.query_biases(text, n_results=5)
        system_prompt = self._create_system_prompt(relevant_biases)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Проанализируй: \"{text}\"")
        ]

        try:
            res = self.llm.invoke(messages)
            raw_content = res.content

            # Очистка JSON от markdown
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()

            analysis_data = json.loads(raw_content)

            # Верификация
            if "cognitive_biases" in analysis_data:
                verified = [b for b in analysis_data["cognitive_biases"] if self._verify_bias(text, b.get("name"))]
                analysis_data["cognitive_biases"] = verified

            return analysis_data

        except Exception as e:
            logging.error(f"Ошибка анализа: {e}")
            return default_response
