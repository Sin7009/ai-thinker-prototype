import unittest
from unittest.mock import MagicMock, patch
import json
from orchestrator.orchestrator import Orchestrator, AgentMode, PartnerState
from agents.bias_mapping import RUSSIAN_TO_INTERNAL_BIAS_MAP

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом."""
        self.orchestrator = Orchestrator(user_id_stub="test_user")
        # Мокаем (имитируем) внешние зависимости
        self.orchestrator.task_agent = MagicMock()
        self.orchestrator.methodology_agent = MagicMock()
        self.orchestrator.detector_agent = MagicMock()
        self.orchestrator.action_library = MagicMock()
        self.orchestrator.memory = MagicMock()

        # Мокаем метод вывода черт, чтобы он не мешал другим тестам
        self.orchestrator._infer_and_save_user_traits = MagicMock()

        # По умолчанию detector_agent.analyze возвращает пустой JSON-список
        self.orchestrator.detector_agent.analyze.return_value = json.dumps([])

    def test_initial_state(self):
        """Тест: начальное состояние Оркестратора."""
        self.assertEqual(self.orchestrator.mode, AgentMode.COPILOT)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.IDLE)

    def test_switch_to_partner_mode(self):
        """Тест: переключение в режим Партнера."""
        self.orchestrator.switch_mode(AgentMode.PARTNER)
        self.assertEqual(self.orchestrator.mode, AgentMode.PARTNER)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.IDLE)

        response = self.orchestrator.process_input("Хочу начать")
        self.assertIn("опишите проблему или ситуацию", response)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.AWAITING_PROBLEM)

    def test_partner_mode_flow(self):
        """Тест: полный цикл прохождения по состояниям в режиме Партнера."""
        self.orchestrator.switch_mode(AgentMode.PARTNER)

        # 1. Начало -> AWAITING_PROBLEM
        self.orchestrator.process_input("Начинаем")
        self.assertEqual(self.orchestrator.partner_state, PartnerState.AWAITING_PROBLEM)

        # 2. Описание проблемы -> DECONSTRUCTION
        self.orchestrator.methodology_agent.execute.return_value = "Ответ деконструкции"
        response = self.orchestrator.process_input("Моя проблема в том, что все плохо")
        self.assertEqual(self.orchestrator.partner_state, PartnerState.DECONSTRUCTION)
        self.orchestrator.methodology_agent.execute.assert_called_once()
        self.assertIn("Ответ деконструкции", response)

        # 3. "дальше" -> HYPOTHESIS_FIELD
        self.orchestrator.methodology_agent.reset_mock()
        self.orchestrator.methodology_agent.execute.return_value = "Ответ генерации гипотез"
        response = self.orchestrator.process_input("дальше")
        self.assertEqual(self.orchestrator.partner_state, PartnerState.HYPOTHESIS_FIELD)
        self.orchestrator.methodology_agent.execute.assert_called_once()
        self.assertIn("Ответ генерации гипотез", response)

    def test_new_detector_agent_integration(self):
        """
        Тест: Оркестратор правильно обрабатывает ответ от нового DetectorAgent.
        """
        # 1. Настраиваем мок DetectorAgent
        mock_detector_response = [
            {
                "name": "Катастрофизация",
                "confidence": 95,
                "context": "Пользователь преувеличивает последствия."
            },
            {
                "name": "Неизвестное искажение", # Этого искажения нет в карте
                "confidence": 80,
                "context": "Какой-то текст."
            }
        ]
        self.orchestrator.detector_agent.analyze.return_value = json.dumps(mock_detector_response)

        # 2. Мокаем _should_propose_partner_mode, чтобы проверить, с какими данными он вызывается
        with patch.object(self.orchestrator, '_should_propose_partner_mode', return_value=(False, "", None)) as mock_propose:
            # 3. Вызываем тестируемый метод
            self.orchestrator.process_input("Это просто ужас, все пропало.")

            # 4. Проверки
            # Убедимся, что analyze был вызван
            self.orchestrator.detector_agent.analyze.assert_called_once_with("Это просто ужас, все пропало.")

            # Проверим, что save_cognitive_pattern был вызван только для известного искажения
            # и с правильным, внутренним именем 'catastrophizing'
            self.orchestrator.memory.save_cognitive_pattern.assert_called_once_with(
                pattern_name='catastrophizing',
                confidence=95,
                context="Пользователь преувеличивает последствия."
            )

            # Проверим, что _should_propose_partner_mode был вызван с обработанными данными
            # (включая добавленный ключ 'bias')
            mock_propose.assert_called_once()
            call_args = mock_propose.call_args[0][0] # Получаем первый аргумент вызова
            self.assertEqual(len(call_args), 1)
            self.assertEqual(call_args[0]['bias'], 'catastrophizing')
            self.assertEqual(call_args[0]['name'], 'Катастрофизация')


    def test_reset_partner_session(self):
        """Тест: сброс сессии в режиме Партнера."""
        self.orchestrator.switch_mode(AgentMode.PARTNER)
        self.orchestrator.process_input("Начать")
        self.orchestrator.process_input("Проблема")

        self.assertNotEqual(self.orchestrator.partner_state, PartnerState.IDLE)

        self.orchestrator.reset_partner_session()
        self.assertEqual(self.orchestrator.partner_state, PartnerState.IDLE)
        self.assertIsNone(self.orchestrator.last_partner_result)

    def test_enrich_context(self):
        """Тест: обогащение контекста данными из памяти."""
        # Настраиваем моки
        self.orchestrator.memory.search_memories.return_value = ["старый диалог"]
        self.orchestrator.memory.get_user_profile_summary.return_value = "Профиль пользователя"

        context = self.orchestrator._enrich_context("новый запрос")

        # Проверяем, что методы были вызваны
        self.orchestrator.memory.search_memories.assert_called_with("новый запрос", n_results=3)
        self.orchestrator.memory.get_user_profile_summary.assert_called_once()

        # Проверяем содержимое
        self.assertIn("Профиль пользователя", context)
        self.assertIn("старый диалог", context)

    def test_infer_and_save_user_traits(self):
        """Тест: вывод и сохранение черт пользователя."""
        # Возвращаем мок к исходному состоянию
        self.orchestrator._infer_and_save_user_traits.reset_mock()
        # "Размокаем" метод для этого конкретного теста
        self.orchestrator._infer_and_save_user_traits = Orchestrator._infer_and_save_user_traits.__get__(self.orchestrator)

        # Настраиваем моки
        mock_response = '[{"trait_type": "interest", "trait_description": "AI", "confidence": 80}]'
        self.orchestrator.task_agent.process.return_value = mock_response

        self.orchestrator._infer_and_save_user_traits("ввод", "ответ")

        # Проверяем, что был вызван метод сохранения
        self.orchestrator.memory.save_user_trait.assert_called_once_with(
            trait_type="interest",
            trait_description="AI",
            confidence=80
        )


class TestNameAndGreeting(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом."""
        self.orchestrator = Orchestrator(user_id_stub="test_user_greeting")
        self.orchestrator.memory = MagicMock()

    def test_extract_name_positive(self):
        """Тест: _extract_name корректно извлекает имена."""
        self.assertEqual(self.orchestrator._extract_name("Меня зовут Алексей"), "Алексей")
        self.assertEqual(self.orchestrator._extract_name("Я — Мария"), "Мария")
        self.assertEqual(self.orchestrator._extract_name("  Екатерина  "), "Екатерина")
        self.assertEqual(self.orchestrator._extract_name("это Пётр"), "Пётр")

    def test_extract_name_negative(self):
        """Тест: _extract_name игнорирует приветствия и некорректные фразы."""
        self.assertIsNone(self.orchestrator._extract_name("Привет"))
        self.assertIsNone(self.orchestrator._extract_name("  добрый день "))
        self.assertIsNone(self.orchestrator._extract_name("Как дела?"))
        self.assertIsNone(self.orchestrator._extract_name("я хочу поговорить"))
        self.assertIsNone(self.orchestrator._extract_name("алексей")) # Должно быть с большой буквы

    def test_get_greeting_new_user(self):
        """Тест: get_greeting для нового пользователя."""
        self.orchestrator.memory.get_user_name.return_value = None
        self.orchestrator.memory.get_last_session_summary.return_value = None
        greeting = self.orchestrator.get_greeting()
        self.assertIn("Здравствуйте!", greeting)
        self.assertIn("как я могу к вам обращаться?", greeting)

    def test_get_greeting_returning_user_no_summary(self):
        """Тест: get_greeting для вернувшегося пользователя без саммари."""
        self.orchestrator.memory.get_user_name.return_value = "Иван"
        self.orchestrator.memory.get_last_session_summary.return_value = None
        greeting = self.orchestrator.get_greeting()
        self.assertIn("Иван, рад вас снова видеть!", greeting)
        self.assertIn("Чем я могу вам помочь сегодня?", greeting)

    def test_get_greeting_returning_user_with_summary(self):
        """Тест: get_greeting для вернувшегося пользователя с саммари."""
        self.orchestrator.memory.get_user_name.return_value = "Анна"
        self.orchestrator.memory.get_last_session_summary.return_value = "обсуждали котиков"
        greeting = self.orchestrator.get_greeting()
        self.assertIn("Анна, рад вас снова видеть!", greeting)
        self.assertIn("обсуждали котиков", greeting)


if __name__ == '__main__':
    unittest.main()
