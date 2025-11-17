import uuid
import json
from agents.task_agent import TaskAgent
from agents.detector_agent import DetectorAgent
from agents.methodology_agent import MethodologyAgent  # ← Добавлено
from orchestrator.dynamic_memory import DynamicMemory
from orchestrator.action_library import ActionLibrary #Нужно реализовать библиотеку действий
from database.db_connector import get_chroma_collection, chroma_client
import re

from langchain_core.messages import HumanMessage, SystemMessage  # ← Добавлено!

from .agent_mode import AgentMode
from .partner_state import PartnerState

class Orchestrator:
    def __init__(self, user_id_stub: str):
        self.user_id_stub = user_id_stub
        self.memory = DynamicMemory(user_id_stub)
        self.task_agent = TaskAgent()
        self.methodology_agent = MethodologyAgent()
        self.detector_agent = DetectorAgent()
        self.mode = AgentMode.COPILOT
        self.partner_state = PartnerState.IDLE  # ← Должно быть именно так
        self.last_partner_result = None
        self.partnership_proposed = False
        self.last_user_input = ""
        self.vector_collection = get_chroma_collection(f"dialogue_vector_{user_id_stub}")
        self.action_library = ActionLibrary()
        print(f"Оркестратор инициализирован для пользователя {user_id_stub}.")


    def _extract_name(self, text: str) -> str:
        """
        Пытается извлечь имя из фраз: "Меня зовут Костя", "Я — Костя", "Костя".
        Возвращает имя или None.
        """
        text = text.strip()
        if not text:
            return None

        # Паттерн 1: "зовут Костя", "я — Костя", "это Костя"
        match = re.search(
            r"(?:зовут|это|я[^\w]*|меня зовут)\s+([А-ЯЁ][а-яё]+)",
            text,
            re.IGNORECASE
        )
        if match:
            return match.group(1)  # ← Только одна группа!

        # Паттерн 2: просто имя (одно слово с заглавной)
        if re.fullmatch(r"[А-ЯЁ][а-яё]+", text):
            return text

        return None


    
    def _sync_agent_memories(self):
        """Передаёт известные факты в агентов (опционально)"""
        # Например: если вы храните имя пользователя, можно передать в системный промпт
        pass

    def _should_retrieve_memory(self, text: str) -> bool:
        """Проверяет, нужно ли извлекать память."""
        triggers = [
            "о чём мы говорили", "что было", "напомни", "раньше говорил",
            "прошлый раз", "уже обсуждали", "говорили ли", "помнит", "напомни"
        ]
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in triggers)

    def _should_trigger_auto_analysis(self, detected_patterns: list) -> tuple[bool, str]:
        """
        Проверяет, нужно ли предложить анализ.
        Возвращает (True, паттерн) если нужно.
        """
        if not detected_patterns:
            return False, ""

        # Берём самый частотный паттерн
        for pattern in detected_patterns:
            bias = pattern['bias']
            frequency = self.memory.get_pattern_frequency(bias)
            if frequency >= 2:  # Если уже встречался 2+ раза
                return True, bias

        return False, ""


    def _normalize_text(self, text: str) -> str:
        """Убирает лишние символы, приводит к нижнему регистру."""
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def _should_report_memory(self, text: str) -> bool:
        text_norm = self._normalize_text(text)
        triggers = [
            "расскажи про меня",
            "что ты обо мне знаешь",
            "что ты обо мне помнишь",
            "что ты помнишь",
            "что ты знаешь",
            "напомни",
            "о чём мы говорили",
            "что было",
            "уже обсуждали",
            "что обо мне"
        ]
        return any(trigger in text_norm for trigger in triggers)


    
    def process_input(self, text: str) -> str:
        self.memory.save_interaction(text, is_user=True)

        # Мета-анализ
        analysis_result = self.detector_agent.analyze(text)
        detected_patterns = json.loads(analysis_result)
        for pattern in detected_patterns:
            self.memory.save_cognitive_pattern(
                pattern_name=pattern['bias'],
                confidence=pattern['confidence'],
                context=pattern['context']
            )

        # 🔥 Авто-переключение в режим 'Партнёр'
        if (self.mode == AgentMode.COPILOT and
            self._should_auto_switch_to_partner(detected_patterns) and
            not self.partnership_proposed):

            self.switch_mode(AgentMode.PARTNER)
            self.partnership_proposed = True

            response = (
                "🔍 Я вижу, что тема становится глубже. "
                "Автоматически переключаюсь в режим 'Партнёр' для более глубокой работы.\n\n"
                "Пожалуйста, опишите, что именно вас беспокоит — мы начнём с деконструкции."
            )
            self.memory.save_interaction(response, is_user=False)
            return response

        # 🔍 Проверка: не просит ли пользователь вспомнить
        if self._should_report_memory(text):
            user_summary = self.memory.get_user_profile_summary()
            response = f"Я помню следующее о тебе:\n\n{user_summary}\n\nХочешь углубиться в какую-то тему?"
            self.memory.save_interaction(response, is_user=False)
            return response

        # Основная логика
        if self.mode == AgentMode.COPILOT:
            context_memory = self.memory.get_last_session_summary_for_prompt()
            if self._should_retrieve_memory(text):
                relevant_memories = self.memory.search_memories(text, n_results=3)
                if relevant_memories:
                    context_memory += "\n\n🧠 Из вашего прошлого диалога:\n" + "\n".join([
                        f"- «{m}»" for m in relevant_memories
                    ])
            response = self.handle_copilot_mode(text, detected_patterns, context_memory)
        elif self.mode == AgentMode.PARTNER:
            response = self.handle_partner_mode(text)
        else:
            response = "Ошибка: неизвестный режим работы."

        self.memory.save_interaction(response, is_user=False)

        return response


    def _should_auto_switch_to_partner(self, detected_patterns: list) -> bool:
        """
        Проверяет, нужно ли автоматически переключиться в режим 'Партнёр'.
        Условия: 2+ разных паттерна ИЛИ один паттерн, но уже встречался 2+ раза.
        """
        if len(detected_patterns) == 0:
            return False

        # Условие 1: 2 и более разных паттерна
        unique_biases = {p['bias'] for p in detected_patterns}
        if len(unique_biases) >= 2:
            return True

        # Условие 2: один паттерн, но уже встречался 2+ раза
        for pattern in detected_patterns:
            bias = pattern['bias']
            frequency = self.memory.get_pattern_frequency(bias)
            if frequency >= 2:
                return True

        return False


    def handle_copilot_mode(self, text, detected_patterns, context_memory=""):
        response = self.task_agent.process(text, context_memory=context_memory)

        # 🔍 Авто-анализ: если паттерн уже встречался
        should_trigger, bias = self._should_trigger_auto_analysis(detected_patterns)
        if should_trigger and not self.partnership_proposed:
            bias_names = {
                "black_and_white_thinking": "черно-белое мышление",
                "catastrophizing": "катастрофизация",
                "overgeneralization": "чрезмерное обобщение",
                "personalization": "персонализация"
            }
            readable = bias_names.get(bias, bias)

            response += (
                f"\n\n🔍 Я заметил, что ты снова используешь признаки '{readable}'. "
                "Ты уже упоминал это раньше. "
                "Хочешь перейти в режим 'Партнёр' и глубже разобраться с этим паттерном? "
                "(введите '/partner')"
            )
            self.partnership_proposed = True

        return response



    def handle_partner_mode(self, text: str) -> str:
        """Обрабатывает режим 'Партнёр' с пошаговой деконструкцией."""
        if self.partner_state == PartnerState.IDLE:
            # Начало — ждём проблему
            self.partner_state = PartnerState.AWAITING_PROBLEM
            return (
                "🔍 Отлично, мы в режиме 'Партнёр'.\n"
                "Пожалуйста, опишите, что именно вас беспокоит — "
                "мы начнём с деконструкции вашего запроса."
            )
        if self.partner_state == PartnerState.DECONSTRUCTING:
            result = self.action_library.run_deconstruction(text)
            self.partner_state = PartnerState.REFRAMING
            return result
        
        if self.partner_state == PartnerState.AWAITING_PROBLEM:
            # Сохраняем проблему
            self.last_partner_result = {
                "problem": text,
                "patterns": json.loads(self.detector_agent.analyze(text))
            }
            self.partner_state = PartnerState.DECONSTRUCTING

            # Задаём первый вопрос деконструкции
            return (
                f"Вы сказали: «{text}».\n\n"
                "🔍 Давайте разберём это. Ответьте на три вопроса:\n"
                "1. Что именно имелось в виду под 'всё'?\n"
                "2. Когда вы впервые почувствовали это?\n"
                "3. Что было бы, если бы это не было правдой?"
            )

        if self.partner_state == PartnerState.DECONSTRUCTING:
            # Пока просто собираем ответы (в будущем — анализ)
            self.partner_state = PartnerState.REFRAMING
            return (
                "Спасибо за ответы. Теперь попробуем переформулировать.\n\n"
                "Как бы вы описали эту ситуацию, если бы смотрели на неё со стороны?\n"
                "Попробуйте начать с: «Кажется, что...»"
            )

        if self.partner_state == PartnerState.REFRAMING:
            self.partner_state = PartnerState.STRATEGIZING
            return (
                "Отлично. Теперь — стратегия.\n\n"
                "Что вы могли бы сделать по-другому завтра, чтобы слегка изменить эту ситуацию?"
            )

        if self.partner_state == PartnerState.STRATEGIZING:
            self.partner_state = PartnerState.IDLE
            return (
                "Благодарю за глубокий разбор.\n\n"
                "Вы прошли полный цикл: деконструкция → переосмысление → стратегия.\n"
                "Можете вернуться в любой момент. Готов продолжить — просто скажите."
            )
        if self._is_user_data_challenging_core_belief():
            return (
                "Я заметил, что ваш опыт противоречит моей текущей модели. "
                "Давайте пересмотрим базовые предпосылки — возможно, мне нужно переосмыслить подход?"
            )

        
        # fallback
        return "Режим 'Партнёр': неизвестное состояние. Попробуйте начать сначала."
    
    def switch_mode(self, new_mode: AgentMode):
        self.mode = new_mode
        self.partnership_proposed = False
        if new_mode == AgentMode.PARTNER:
            self.partner_state = PartnerState.IDLE
            self.last_partner_result = None
        if new_mode != AgentMode.PARTNER:
            self.partnership_proposed = False
        print(f"Режим изменен на: {self.mode.value}")

    def reset_partner_session(self):
        self.partner_state = PartnerState.IDLE
        self.last_partner_result = None
        self.methodology_agent.memory.clear()  # ← Очистка памяти агента
        print("Сессия 'Партнер' сброшена.")

    def reset_all_memory(self):
        self.task_agent.clear_memory()
        self.methodology_agent.memory.clear()
        self.partner_state = PartnerState.IDLE
        self.last_partner_result = None
        print("Вся память агентов очищена.")

    def summarize_session(self, last_user_input: str = "") -> str:
        """Генерирует краткое резюме текущей сессии."""
        try:
            summary_prompt = (
                "Ты — AI-методолог. Ниже приведён фрагмент нашего диалога. "
                "Сделай краткое резюме: о чём шла речь, какие ключевые темы, эмоции, выводы. "
                "Не более 2–3 предложений.\n\n"
                f"Последний ввод пользователя: {last_user_input}\n"
                f"Последние несколько реплик (из памяти):\n"
            )

            # Взять последние N сообщений
            recent_messages = self.task_agent.memory.chat_memory.messages[-6:]
            recent_text = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])

            full_prompt = summary_prompt + recent_text

            response = self.task_agent.chat.invoke([HumanMessage(content=full_prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Ошибка при генерации резюме: {e}")
            return "Сессия была посвящена обсуждению личных и профессиональных вызовов."

    # orchestrator/orchestrator.py

    def end_session(self):
        """Генерирует резюме и анализирует прогресс по ВСЕМ когнитивным паттернам."""
        try:
            messages = self.task_agent.memory.chat_memory.messages
            if not messages:
                summary = "Обсуждались общие темы."
            else:
                recent = messages[-6:]
                context = "\n".join([f"{m.type}: {m.content}" for m in recent])
                prompt = (
                    "Сделай краткое резюме нашего диалога (2–3 предложения). "
                    "О чём шла речь? Какие темы, эмоции, выводы? "
                    "Говори от третьего лица, без 'пользователь сказал'. "
                    "Не используй шаблоны. Будь точен.\n\n"
                    f"Последние реплики:\n{context}"
                )
                response = self.task_agent.chat.invoke([HumanMessage(content=prompt)])
                summary = response.content.strip()

            # 🔍 Анализ прогресса по ВСЕМ паттернам
            print("📊 АНАЛИЗ КОГНИТИВНЫХ ПАТТЕРНОВ:")
            active_patterns = [
                "black_and_white_thinking",
                "overgeneralization",
                "catastrophizing",
                "mind_reading",
                "personalization",
                "emotional_reasoning",
                "hindsight_bias",
                "availability_heuristic",
                "status_quo_bias",
                "gamblers_fallacy",
                "survivorship_bias",
                "false_consensus_effect",
                "halo_effect"
            ]

            for pattern_name in active_patterns:
                weight = self.memory.get_pattern_weight_over_time(pattern_name, window_days=30)
                if weight > 0:
                    bias_names = {
                        "black_and_white_thinking": "Черно-белое мышление",
                        "overgeneralization": "Сверхобобщение",
                        "catastrophizing": "Катастрофизация",
                        "mind_reading": "Чтение мыслей",
                        "personalization": "Персонализация",
                        "emotional_reasoning": "Эмоциональное обоснование",
                        "hindsight_bias": "Ошибка ретроспективного взгляда",
                        "availability_heuristic": "Эвристика доступности",
                        "status_quo_bias": "Отклонение в сторону статуса кво",
                        "gamblers_fallacy": "Ошибка игрока",
                        "survivorship_bias": "Ошибка выжившего",
                        "false_consensus_effect": "Эффект ложного консенсуса",
                        "halo_effect": "Эффект ореола"
                    }
                    readable = bias_names.get(pattern_name, pattern_name)
                    print(f"• {readable}: {weight}")

                    # Прогресс
                    if weight < 1.5:
                        print(f"  ✅ Снижение — пользователь прогрессирует.")
                    elif weight > 4.0:
                        print(f"  ⚠️ Высокая частота — требуется внимание.")
                    else:
                        print(f"  🔁 Паттерн сохраняется — продолжаем работу.")

            self.memory.save_session_summary(summary)
        except Exception as e:
            print(f"Ошибка при резюмировании: {e}")
            self.memory.save_session_summary("Обсуждались личные и профессиональные темы.")



