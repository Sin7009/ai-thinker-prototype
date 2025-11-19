import uuid
import json
import threading
from agents.task_agent import TaskAgent
from agents.detector_agent import DetectorAgent
from agents.methodology_agent import MethodologyAgent
from agents.bias_mapping import RUSSIAN_TO_INTERNAL_BIAS_MAP
from orchestrator.dynamic_memory import DynamicMemory
from orchestrator.action_library import ActionLibrary #Нужно реализовать библиотеку действий
from database.db_connector import get_chroma_collection, chroma_client
import re

import time

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_mode import AgentMode

# Актуальные модели второго поколения
MODEL_LITE = "GigaChat-2"
MODEL_SMART = "GigaChat-2-Pro"
MODEL_MAX = "GigaChat-2-Max"

class Orchestrator:
    def __init__(self, user_id_stub: str):
        self.user_id_stub = user_id_stub

        # --- РОУТИНГ МОДЕЛЕЙ ---
        self.task_agent = TaskAgent(model_name=MODEL_LITE)
        self.detector_agent = DetectorAgent(model_name=MODEL_LITE)
        self.methodology_agent = MethodologyAgent(user_id=user_id_stub, model_name=MODEL_SMART)

        self.memory = DynamicMemory(user_id_stub, self.task_agent)
        self.mode = AgentMode.COPILOT
        self.last_user_input = ""
        self.vector_collection = get_chroma_collection(f"dialogue_vector_{user_id_stub}")
        self.action_library = ActionLibrary(self.methodology_agent)
        self.strategic_note = "" # Здесь будет храниться стратегия на сессию
        print(f"Оркестратор инициализирован для пользователя {user_id_stub}.")
        self._develop_strategy() # Вырабатываем стратегию при старте

    def _develop_strategy(self):
        """
        Анализирует историю сессий и формирует 'стратегическую заметку'
        для улучшения качества ответов в текущей сессии.
        """
        recent_analyses = self.memory.get_recent_session_analyses(limit=5)
        if not recent_analyses:
            return # Стратегию не вырабатываем, если истории нет

        history_summary = "\n".join(
            [f"- Сессия от {a.ended_at.strftime('%Y-%m-%d')}: "
             f"Темы ({a.key_topics}), Паттерны ({a.identified_patterns}). "
             f"Резюме: {a.session_summary}" for a in recent_analyses]
        )

        strategy_prompt = f"""
Ты — AI-стратег. Проанализируй историю сессий пользователя и дай короткую (1-2 предложения) тактическую рекомендацию для AI-ассистента на следующую сессию.

История сессий:
{history_summary}

Пример рекомендации: "Пользователь часто возвращается к теме прокрастинации, но техники не помогают. В этот раз стоит попробовать обсудить его эмоции, а не искать решения."
Твоя рекомендация:
"""

        try:
            self.strategic_note = self.task_agent.process("", context_memory=strategy_prompt)
            print(f"💡 Стратегическая заметка на сессию: {self.strategic_note}")
        except Exception as e:
            print(f"Ошибка при разработке стратегии: {e}")

    def _extract_name(self, text: str) -> str:
        """
        Пытается извлечь имя из фраз: "Меня зовут Костя", "Я — Костя", "Костя".
        Возвращает имя или None. Игнорирует распространенные приветствия.
        """
        text_clean = text.strip()
        if not text_clean:
            return None

        # Список слов для исключения (в нижнем регистре)
        greetings = ["привет", "здравствуй", "здравствуйте", "добрый день", "доброе утро", "добрый вечер"]
        if text_clean.lower() in greetings:
            return None

        # Паттерн 1: "зовут Костя", "я — Костя", "это Костя"
        match = re.search(
            # (?i:...) делает часть выражения нечувствительной к регистру,
            # в то время как имя ([А-ЯЁ][а-яё]+) остается чувствительным.
            r"(?i:зовут|это|я[^\w]*|меня зовут)\s+([А-ЯЁ][а-яё]+)",
            text_clean
        )
        if match:
            return match.group(1)

        # Паттерн 2: просто имя (одно слово с заглавной, не из списка приветствий)
        if re.fullmatch(r"[А-ЯЁ][а-яё]+", text_clean):
            return text_clean

        return None

    def get_greeting(self) -> str:
        """
        Генерирует приветствие в зависимости от того, новый ли это пользователь.
        """
        user_name = self.memory.get_user_name()
        last_summary = self.memory.get_last_session_summary()

        if user_name:
            greeting = f"{user_name}, рад вас снова видеть! "
            if last_summary:
                greeting += f"В прошлый раз мы говорили о следующем: '{last_summary}'. Хотите продолжить или у вас новая задача?"
            else:
                greeting += "Чем я могу вам помочь сегодня?"
        else:
            greeting = "Здравствуйте! Чтобы наш диалог был продуктивнее, скажите, как я могу к вам обращаться?"

        return greeting
    
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

    def _should_enter_thinking_cycle(self, text: str) -> bool:
        """
        Определяет, нужно ли переходить в режим "Партнёр" (мыслительный цикл)
        на основе ключевых фраз пользователя.
        """
        triggers = [
            "давай подумаем", "помоги решить", "что мне делать",
            "не могу понять", "нужен совет", "помоги разобраться"
        ]
        text_lower = text.lower().strip()
        return any(trigger in text_lower for trigger in triggers)

    def _diagnose_and_select_action(self, problem_description: str) -> callable:
        """
        Использует LLM для анализа проблемы и выбора наилучшего действия
        из ActionLibrary.
        """
        system_prompt = (
            "Ты — AI-диагност. Твоя задача — проанализировать запрос пользователя и выбрать "
            "наиболее подходящую мыслительную технику для его решения. "
            "Вот доступные тебе инструменты: "
            "1. 'run_rubber_duck_debugging': Используй, когда пользователь застрял в технической проблеме, "
            "баге в коде или не может ясно сформулировать последовательность действий. Идеально для дебаггинга. "
            "2. 'run_five_whys': Используй, когда проблема кажется поверхностной, и нужно докопаться до "
            "глубинной, корневой причины. Отлично подходит для организационных или личных проблем. "
            "3. 'run_constrained_brainstorming': Используй, когда пользователь жалуется на отсутствие идей, "
            "творческий ступор или 'паралич чистого листа'. "
            "В ответ ты должен вернуть ТОЛЬКО название функции, которую нужно вызвать. Например: 'run_five_whys'."
        )

        # Мы используем TaskAgent как "мозг" для этой задачи
        raw_response = self.task_agent.process(problem_description, context_memory=system_prompt)

        # Извлекаем название функции из ответа
        action_name = raw_response.strip()

        # Получаем саму функцию из ActionLibrary
        action_function = getattr(self.action_library, action_name, None)

        if action_function and callable(action_function):
            return action_function
        else:
            # Если LLM вернул что-то не то, используем "утенка" по умолчанию
            return self.action_library.run_rubber_duck_debugging

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


    
    def _run_analysis_in_background(self, text: str):
        """
        Запускает психолингвистический анализ в фоновом потоке
        и сохраняет результаты в базу данных.
        """

        # Ждем 3 секунды, чтобы основной TaskAgent успел отработать
        # и не создавать пиковую нагрузку на API
        time.sleep(3)
        try:
            analysis_data = self.detector_agent.analyze(text)
            if 'cognitive_biases' in analysis_data and isinstance(analysis_data.get('cognitive_biases'), list):
                for pattern in analysis_data['cognitive_biases']:
                    internal_name = RUSSIAN_TO_INTERNAL_BIAS_MAP.get(pattern.get('name'))
                    if internal_name:
                        self.memory.save_cognitive_pattern(
                            pattern_name=internal_name,
                            confidence=pattern.get('confidence', 0),
                            context=pattern.get('context', '')
                        )
            self.memory.save_psycholinguistic_features(
                emotional_tone=analysis_data.get('emotional_tone', 'Нейтральный'),
                communication_style=analysis_data.get('communication_style', 'Аналитический')
            )
        except Exception as e:
            print(f"Ошибка в фоновом потоке анализа: {e}")

    def process_input(self, text: str) -> str:
        self.memory.save_interaction(text, is_user=True)
        self.last_user_input = text

        # 🚀 **Новый пайплайн обработки** 🚀

        # 1. Асинхронный психолингвистический анализ (Fire-and-Forget)
        if len(text.split()) > 7:  # Порог на минимальную длину сообщения
            analysis_thread = threading.Thread(target=self._run_analysis_in_background, args=(text,))
            analysis_thread.start()

        # 2. Проверка на запрос о памяти
        if self._should_report_memory(text):
            user_summary = self.memory.get_user_profile_summary()
            response = f"Я помню следующее о тебе:\n\n{user_summary}"
            self.memory.save_interaction(response, is_user=False)
            return response

        # 3. Проверка на вход в мыслительный цикл
        if self._should_enter_thinking_cycle(text):
            self.switch_mode(AgentMode.PARTNER)
            response = self.handle_partner_mode(text)
            self.memory.save_interaction(response, is_user=False)
            return response

        # 4. Основная логика по режимам
        if self.mode == AgentMode.COPILOT:
            response = self.handle_copilot_mode(text)
        elif self.mode == AgentMode.PARTNER:
            response = self.handle_partner_mode(text)
            # ПРОВЕРКА НА ВЫХОД ИЗ ТЕХНИКИ
            if "[STOP_TECHNIQUE]" in response:
                self.switch_mode(AgentMode.COPILOT)
                response = "Хорошо, без проблем. Возвращаемся в обычный режим. Чем еще могу помочь?"
        else:
            response = "Ошибка: неизвестный режим работы."

        # 5. Сохранение и вывод
        self.memory.save_interaction(response, is_user=False)
        self._infer_and_save_user_traits(text, response)
        return response

    def _infer_and_save_user_traits(self, user_input: str, agent_response: str):
        """
        Анализирует последний обмен сообщениями, чтобы вывести и сохранить
        черты пользователя (предпочтения, интересы и т.д.).
        """
        system_prompt = (
            "Ты — AI-аналитик, специализирующийся на психологии. Твоя задача — "
            "проанализировать диалог и сделать выводы о пользователе. "
            "Основывайся только на предоставленном тексте. "
            "Верни свои выводы в виде списка JSON-объектов. Каждый объект должен иметь "
            "три ключа: 'trait_type' (тип черты: 'preference', 'interest', 'communication_style'), "
            "'trait_description' (описание черты) и 'confidence' (твоя уверенность в выводе от 0 до 100). "
            "Если выводов нет, верни пустой список []."
        )

        # Мы используем TaskAgent как "мозг" для этой задачи
        # В будущем это может быть отдельный, специализированный агент
        dialogue_snippet = f"Пользователь: «{user_input}»\nАгент: «{agent_response}»"

        try:
            # Используем TaskAgent для вывода
            raw_response = self.task_agent.process(dialogue_snippet, context_memory=system_prompt)

            # Извлекаем JSON из ответа
            json_part = raw_response[raw_response.find('['):raw_response.rfind(']')+1]
            inferred_traits = json.loads(json_part)

            for trait in inferred_traits:
                if all(k in trait for k in ['trait_type', 'trait_description', 'confidence']):
                    # Пониженный порог для создания гипотезы
                    if trait['confidence'] > 50:
                        self.memory.reinforce_user_trait(
                            trait_type=trait['trait_type'],
                            trait_description=trait['trait_description'],
                            confidence=trait['confidence']
                        )
        except (json.JSONDecodeError, IndexError) as e:
            # Ошибки парсинга JSON — это нормально, если LLM ответил не в том формате
            # print(f"Не удалось извлечь черты из ответа: {raw_response}. Ошибка: {e}")
            pass
        except Exception as e:
            print(f"Произошла ошибка при выводе черт пользователя: {e}")


    def _enrich_context(self, query: str) -> str:
        """
        Собирает и обогащает контекст для передачи в LLM.
        Включает стратегическую заметку, релевантные диалоги (RAG) и сводку профиля.
        """
        full_context = ""

        # 1. Стратегическая заметка (если есть)
        if self.strategic_note:
            full_context += f"**Тактическая рекомендация на эту сессию:** {self.strategic_note}\n\n"

        # 2. RAG из ChromaDB
        relevant_memories = self.memory.search_memories(query, n_results=3)
        rag_context = ""
        if relevant_memories:
            rag_context = "Вот релевантные фрагменты из прошлых диалогов:\n" + "\n".join(
                [f"- «{m}»" for m in relevant_memories]
            )

        # 3. Сводка из SQLite
        profile_summary = self.memory.get_user_profile_summary()

        # 4. Объединение
        if profile_summary:
            full_context += f"**Информация о пользователе:**\n{profile_summary}\n\n"
        if rag_context:
            full_context += f"**Контекст диалога:**\n{rag_context}\n\n"

        return full_context

    def handle_copilot_mode(self, text: str) -> str:
        """
        Обрабатывает режим "Копилот": прямой ответ на запрос пользователя.
        """
        # Обогащаем контекст, чтобы дать LLM больше информации
        enriched_context = self._enrich_context(text)

        # Получаем прямой ответ от TaskAgent
        response = self.task_agent.process(text, context_memory=enriched_context)
        return response

    def handle_partner_mode(self, text: str) -> str:
        """
        Обрабатывает режим "Партнёр": запускает мыслительный цикл.
        Диагностирует проблему и выбирает подходящую технику из ActionLibrary.
        """
        # 1. Диагностируем проблему и выбираем действие
        action_to_run = self._diagnose_and_select_action(text)

        # 2. Запускаем выбранное действие
        # Метод из ActionLibrary сам вызовет MethodologyAgent с нужным промптом
        response = action_to_run(text)

        return response
    
    def switch_mode(self, new_mode: AgentMode):
        """Переключает режим работы Оркестратора."""
        self.mode = new_mode
        if new_mode == AgentMode.COPILOT:
            # При выходе из режима "Партнер" можно очистить память агента методологий
            self.methodology_agent.clear_memory()
            print("Режим изменен на: COPILOT. Сессия партнёрства завершена.")
        else:
            print(f"Режим изменен на: {self.mode.value}.")

    def reset_all_memory(self):
        """Сбрасывает всю память, включая TaskAgent и MethodologyAgent."""
        self.task_agent.clear_memory()
        self.methodology_agent.clear_memory()
        print("Вся память агентов очищена.")

    def _analyze_and_save_session(self):
        """
        Проводит глубокий анализ завершенной сессии, извлекает ключевые темы и паттерны,
        и сохраняет результат в базу данных.
        """
        # 1. Получаем историю диалога
        messages = self.task_agent.memory.chat_memory.messages
        if len(messages) < 4:
            print("Недостаточно сообщений для анализа сессии.")
            return

        dialogue_history = "\n".join([f"{m.type}: {m.content}" for m in messages])

        # 2. Формулируем промпт для LLM
        analysis_prompt = f"""
Ты — AI-аналитик. Проанализируй следующий диалог и верни СТРОГО JSON-объект со следующими ключами:
- "session_summary": Краткое резюме диалога в 2-3 предложениях.
- "key_topics": Список из 3-5 ключевых тем или слов, обсуждавшихся в диалоге (например, ["прокрастинация", "python", "тревожность"]).
- "identified_patterns": Список внутренних названий когнитивных искажений, которые были замечены (например, ["catastrophizing", "overgeneralization"]).

Диалог для анализа:
{dialogue_history}
"""

        try:
            # 3. Вызываем LLM и парсим ответ
            raw_response = self.task_agent.process("", context_memory=analysis_prompt)

            # --- ФИКС: Очистка JSON ---
            # Ищем, где начинается первая { и где заканчивается последняя }
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Вырезаем только JSON-объект, игнорируя текст до и после
                clean_json = raw_response[start_idx : end_idx + 1]
                analysis_result = json.loads(clean_json)
            else:
                # Если скобки не найдены, логируем ошибку и продолжаем
                raise json.JSONDecodeError("Не удалось найти чистый JSON-объект в ответе LLM.", raw_response, 0)
            # --- КОНЕЦ ФИКСА ---

            # 4. Сохраняем результат в базу
            self.memory.save_session_analysis(
                summary=analysis_result.get("session_summary", "Не удалось сгенерировать резюме."),
                topics=analysis_result.get("key_topics", []),
                patterns=analysis_result.get("identified_patterns", [])
            )
            print("Анализ сессии успешно сохранен.")
            self._report_cognitive_patterns()

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Ошибка парсинга JSON при анализе сессии: {e}. Ответ LLM: {raw_response}")
        except Exception as e:
            print(f"Неожиданная ошибка при анализе сессии: {e}")

    def _report_cognitive_patterns(self):
        """
        Выводит в консоль отчет о динамике когнитивных паттернов.
        """
        print("\n📊 АНАЛИЗ КОГНИТИВНЫХ ПАТТЕРНОВ (динамика за 30 дней):")

        # Получаем все уникальные паттерны, которые когда-либо наблюдались у пользователя
        all_patterns = self.memory.get_user_patterns()
        unique_pattern_names = sorted(list({p.pattern_name for p in all_patterns}))

        if not unique_pattern_names:
            print("Паттерны пока не наблюдались.")
            return

        for pattern_name in unique_pattern_names:
            weight = self.memory.get_pattern_weight_over_time(pattern_name, window_days=30)
            if weight > 0:
                # Получаем человекочитаемое имя
                readable_name = next((rus_name for rus_name, internal_name in RUSSIAN_TO_INTERNAL_BIAS_MAP.items() if internal_name == pattern_name), pattern_name)

                print(f"  • {readable_name}: {weight}")
                if weight < 1.5:
                    print("    ✅ Снижение — пользователь прогрессирует.")
                elif weight > 4.0:
                    print("    ⚠️ Высокая частота — требуется внимание.")
                else:
                    print("    🔁 Паттерн сохраняется — продолжаем работу.")

    def end_session(self):
        """
        Публичный метод для корректного завершения сессии.
        Вызывается из main.py при штатном выходе или Ctrl+C.
        """
        print("\nЗавершение работы... Сохранение данных сессии.")
        self._analyze_and_save_session()


