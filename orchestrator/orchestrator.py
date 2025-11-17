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
        self.methodology_agent = MethodologyAgent(user_id=user_id_stub)
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

    def _should_propose_partner_mode(self, detected_patterns: list) -> tuple[bool, str, PartnerState | None]:
        """
        Проверяет, нужно ли предложить переход в режим 'Партнёр'.
        Возвращает (True, причина, рекомендуемый_стейт) если нужно.
        """
        if not detected_patterns:
            return False, "", None

        # Словарь соответствия когнитивных искажений и модулей
        bias_to_module = {
            "black_and_white_thinking": (PartnerState.HYPOTHESIS_FIELD, "увидеть альтернативы"),
            "overgeneralization": (PartnerState.DECONSTRUCTION, "разобрать конкретные факты"),
            "catastrophizing": (PartnerState.STRESS_TESTING, "проверить худшие сценарии"),
            "personalization": (PartnerState.DECONSTRUCTION, "отделить факты от личной ответственности")
        }

        # Условие 2: один паттерн, но уже встречался 2+ раза
        for pattern in detected_patterns:
            bias = pattern['bias']
            frequency = self.memory.get_pattern_frequency(bias)
            if frequency >= 2 and bias in bias_to_module:
                state, reason_text = bias_to_module[bias]
                return True, f"я заметил паттерн '{bias.replace('_', ' ')}' и думаю, мы могли бы {reason_text}", state

        # Условие 1 (фоллбэк): 2 и более разных паттерна
        unique_biases = {p['bias'] for p in detected_patterns}
        if len(unique_biases) >= 2:
            return True, "я обнаружил несколько паттернов мышления, и было бы полезно их распутать", PartnerState.DECONSTRUCTION

        return False, "", None


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


    def handle_copilot_mode(self, text, detected_patterns, context_memory=""):
        # Проверяем, не является ли ответ согласием на предыдущее предложение
        positive_responses = ['да', 'давай', 'хорошо', 'согласен', 'ок']
        if self.partnership_proposed and text.lower().strip() in positive_responses:
            self.switch_mode(AgentMode.PARTNER, start_state=self.proposed_partner_state)
            # Запускаем предложенный модуль с последним вводом пользователя
            return self._run_partner_module(self.partner_state, self.last_user_input)

        # Сначала получаем основной ответ
        response = self.task_agent.process(text, context_memory=context_memory)
        self.last_user_input = text # Сохраняем ввод для возможного перехода

        # Затем, если нужно, добавляем предложение о партнерстве
        should_propose, reason, proposed_state = self._should_propose_partner_mode(detected_patterns)
        if should_propose and not self.partnership_proposed:
            self.partnership_proposed = True
            self.proposed_partner_state = proposed_state  # Сохраняем, какой стейт предложить
            proposal = (
                f"\n\n🔍 Кстати, {reason}. "
                "Это может быть хорошей точкой для более глубокого анализа. "
                "Хотите, мы вместе исследуем эту тему в режиме 'Партнёр'? "
                "Просто скажите 'да', и мы начнём."
            )
            response += proposal

        return response



    def handle_partner_mode(self, text: str) -> str:
        """
        Обрабатывает режим 'Партнёр' как стейт-машину, проводя пользователя
        через полный методологический цикл.
        """
        # Ключевые слова для перехода на следующий этап
        continue_keywords = ['продолжим', 'дальше', 'готовы', 'давай', 'ок', 'хорошо']
        text_lower = text.lower().strip()

        # 1. Начало работы или запрос на новый цикл
        if self.partner_state == PartnerState.IDLE:
            self.partner_state = PartnerState.AWAITING_PROBLEM
            return (
                "🔍 Отлично, мы в режиме 'Партнёр'.\n"
                "Чтобы начать полный цикл анализа, пожалуйста, опишите проблему или ситуацию, "
                "которую вы хотели бы разобрать."
            )

        # 2. Получение проблемы и переход к деконструкции
        if self.partner_state == PartnerState.AWAITING_PROBLEM:
            self.last_partner_result = {"problem": text} # Сохраняем проблему
            self.partner_state = PartnerState.DECONSTRUCTION
            # Запускаем деконструкцию с исходным текстом проблемы
            return self._run_partner_module(PartnerState.DECONSTRUCTION, text)

        # 3. Проверяем, хочет ли пользователь перейти на следующий этап
        if any(keyword in text_lower for keyword in continue_keywords):
            next_state = self._get_next_state(self.partner_state)
            if next_state:
                self.partner_state = next_state
                # Запускаем следующий модуль, передавая ему накопленный контекст
                return self._run_partner_module(next_state, self.last_partner_result.get("problem", text))
            else:
                self.partner_state = PartnerState.IDLE
                return "Цикл завершен. Спасибо за работу! Мы можем начать новый разбор, если хотите."

        # 4. Если не переход, то продолжаем работать в текущем модуле
        return self._run_partner_module(self.partner_state, text)

    def _get_next_state(self, current_state: PartnerState) -> PartnerState | None:
        """Определяет следующий стейт в цикле."""
        order = [
            PartnerState.DECONSTRUCTION,
            PartnerState.HYPOTHESIS_FIELD,
            PartnerState.STRESS_TESTING,
            PartnerState.SYNTHESIS,
            PartnerState.ASSIMILATION
        ]
        try:
            current_index = order.index(current_state)
            if current_index + 1 < len(order):
                return order[current_index + 1]
            return None # Цикл завершен
        except ValueError:
            return None


    def _run_partner_module(self, state: PartnerState, text: str) -> str:
        """Вызывает MethodologyAgent с промптом для конкретного модуля."""
        prompts = {
            PartnerState.DECONSTRUCTION: "Ты — AI-методолог, твоя задача — провести 'деконструкцию' проблемы. Помогай пользователю отделить факты от эмоций и мнений, задавай уточняющие вопросы, чтобы составить ясную 'карту фактов'. Спроси, готовы ли продолжить, когда факты будут собраны.",
            PartnerState.HYPOTHESIS_FIELD: "Ты — AI-методолог. На основе собранных фактов, помоги пользователю сгенерировать 3-4 взаимоисключающие гипотезы. Побуждай к творчеству: очевидная, инвертированная, аналоговая гипотезы. Спроси, готовы ли продолжить, когда гипотезы будут готовы.",
            PartnerState.STRESS_TESTING: "Ты — AI-методолог. Помоги пользователю провести 'стресс-тестинг' выбранной гипотезы. Используй техники 'Pre-mortem' (что если все пойдет не так?), 'Черный лебедь' (поиск фатальной уязвимости). Спроси, готовы ли продолжить.",
            PartnerState.SYNTHESIS: "Ты — AI-методолог. Помоги пользователю синтезировать новую, 'третью идею' из сильных сторон проверенных гипотез. Твоя цель — найти нелинейное, сильное решение. Спроси, готовы ли продолжить.",
            PartnerState.ASSIMILATION: "Ты — AI-методолог. Помоги пользователю 'ассимилировать' новый опыт. Обсудите, как изменилось его понимание проблемы и какие конкретные шаги он может предпринять. Поблагодари за работу."
        }
        system_prompt = prompts.get(state, "Ты — AI-помощник.")

        # Добавляем в промпт накопленный контекст
        full_prompt = (
            f"{system_prompt}\n\n"
            f"**Текущий контекст разбора:**\n"
            f"{self.last_partner_result.get('problem', 'Нет данных')}"
        )

        response = self.methodology_agent.execute(
            system_prompt=full_prompt,
            user_prompt=text
        )

        # Обновляем накопленный результат (очень упрощенно)
        self.last_partner_result["problem"] += f"\n\nОтвет на {state.value}:\n{response}"

        return response
    
    def switch_mode(self, new_mode: AgentMode, start_state: PartnerState | None = None):
        self.mode = new_mode
        self.partnership_proposed = False # Сбрасываем флаг предложения при любой смене режима

        if new_mode == AgentMode.PARTNER:
            # Если передан конкретный стейт, начинаем с него
            if start_state:
                self.partner_state = start_state
                # Инициализируем last_partner_result, чтобы было куда писать
                self.last_partner_result = {"problem": self.last_user_input}
            else:
                self.partner_state = PartnerState.IDLE
                self.last_partner_result = None

        print(f"Режим изменен на: {self.mode.value}. Начальное состояние партнера: {self.partner_state.value}")

    def reset_partner_session(self):
        self.partner_state = PartnerState.IDLE
        self.last_partner_result = None
        self.methodology_agent.clear_memory()  # ← Исправленный вызов
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



