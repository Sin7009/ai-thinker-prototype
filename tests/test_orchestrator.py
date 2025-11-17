import unittest
from unittest.mock import MagicMock
import json
from orchestrator.orchestrator import Orchestrator, AgentMode, PartnerState

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом."""
        self.orchestrator = Orchestrator(user_id_stub="test_user")
        # Мокаем (имитируем) внешние зависимости
        self.orchestrator.task_agent = MagicMock()
        self.orchestrator.methodology_agent = MagicMock() # <-- Добавлен мок
        self.orchestrator.detector_agent = MagicMock()
        self.orchestrator.action_library = MagicMock()
        self.orchestrator.memory = MagicMock()

        # Мокаем новый метод вывода черт, чтобы он не мешал другим тестам
        self.orchestrator._infer_and_save_user_traits = MagicMock()

        # Указываем, что detector_agent.analyze должен возвращать пустой JSON-список
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
        self.assertIn("опишите проблему или ситуацию", response) # <-- Исправленный текст
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


if __name__ == '__main__':
    unittest.main()
