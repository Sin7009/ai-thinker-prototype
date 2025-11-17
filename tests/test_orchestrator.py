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
        self.orchestrator.detector_agent = MagicMock()
        self.orchestrator.action_library = MagicMock()
        self.orchestrator.memory = MagicMock()
        self.orchestrator.history_collection = MagicMock()

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
        self.assertIn("Пожалуйста, опишите проблему", response)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.AWAITING_PROBLEM)

    def test_partner_mode_flow(self):
        """Тест: полный цикл прохождения по состояниям в режиме Партнера."""
        self.orchestrator.switch_mode(AgentMode.PARTNER)

        self.orchestrator.process_input("Начинаем")
        self.assertEqual(self.orchestrator.partner_state, PartnerState.AWAITING_PROBLEM)

        self.orchestrator.action_library.run_deconstruction.return_value = "Факт 1, Факт 2"
        response = self.orchestrator.process_input("Проблема такая-то")
        self.assertIn("Факт 1, Факт 2", response)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.DECONSTRUCTION_COMPLETE)

        self.orchestrator.action_library.run_hypothesis_field.return_value = "Гипотеза А, Гипотеза Б"
        response = self.orchestrator.process_input("да")
        self.assertIn("Гипотеза А, Гипотеза Б", response)
        self.assertEqual(self.orchestrator.partner_state, PartnerState.HYPOTHESIS_COMPLETE)

    def test_reset_partner_session(self):
        """Тест: сброс сессии в режиме Партнера."""
        self.orchestrator.switch_mode(AgentMode.PARTNER)
        self.orchestrator.process_input("Начать")
        self.orchestrator.process_input("Проблема")

        self.assertNotEqual(self.orchestrator.partner_state, PartnerState.IDLE)

        self.orchestrator.reset_partner_session()
        self.assertEqual(self.orchestrator.partner_state, PartnerState.IDLE)
        self.assertIsNone(self.orchestrator.last_partner_result)

if __name__ == '__main__':
    unittest.main()
