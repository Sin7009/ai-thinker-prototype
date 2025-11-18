import unittest
from unittest.mock import MagicMock, patch
from orchestrator.orchestrator import Orchestrator, AgentMode
from orchestrator.action_library import ActionLibrary
from agents.methodology_agent import MethodologyAgent
from database.db_connector import recreate_tables
import json
import time

# Пересоздаем таблицы перед запуском всех тестов
recreate_tables()

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        """
        Настраиваем окружение для тестирования с использованием "обезьяньего патчинга".
        """
        with patch('orchestrator.orchestrator.DynamicMemory') as mock_dynamic_memory:
            # Предотвращаем реальное создание DynamicMemory
            self.orchestrator = Orchestrator(user_id_stub="test_user")
            self.orchestrator.memory = mock_dynamic_memory.return_value

        self.mock_task_agent = MagicMock()
        self.mock_methodology_agent = MagicMock()
        self.mock_detector_agent = MagicMock()
        self.mock_memory = self.orchestrator.memory # Используем замоканный экземпляр

        self.orchestrator.task_agent = self.mock_task_agent
        self.orchestrator.detector_agent = self.mock_detector_agent

        # Мокируем process, чтобы он возвращал валидный JSON-список
        self.mock_task_agent.process.return_value = '[]'

        self.orchestrator.action_library.methodology_agent = self.mock_methodology_agent

    def test_full_thinking_cycle(self):
        """
        Тест: Полный "мыслительный цикл" от ключевой фразы до финального ответа.
        """
        # Настраиваем возвращаемые значения моков
        self.mock_task_agent.process.return_value = 'run_five_whys'
        self.mock_methodology_agent.invoke.return_value = 'Это финальный ответ от техники пяти почему.'
        self.mock_detector_agent.analyze.return_value = "{}"

        # Запускаем процесс
        final_response = self.orchestrator.process_input("Давай подумаем над моей проблемой")

        # Проверяем результаты
        self.assertEqual(self.orchestrator.mode, AgentMode.PARTNER)
        self.mock_task_agent.process.assert_called_once()
        self.mock_methodology_agent.invoke.assert_called_once()
        self.assertEqual(final_response, 'Это финальный ответ от техники пяти почему.')

        # Проверяем детали вызовов
        diagnosis_call = self.mock_task_agent.process.call_args
        self.assertIn("'run_five_whys'", diagnosis_call.kwargs['context_memory'])

        technique_call = self.mock_methodology_agent.invoke.call_args
        self.assertIn("Твоя роль — коуч, использующий технику 'Пять почему'", str(technique_call.args[0]))

    @patch('threading.Thread.start')
    def test_analysis_data_is_parsed_and_saved(self, mock_thread_start):
        """Тест: Оркестратор парсит комплексный JSON и сохраняет все данные."""
        mock_analysis_data = {
            "cognitive_biases": [{"name": "Катастрофизация", "confidence": 90, "context": "Контекст..."}],
            "emotional_tone": "Тревога",
            "communication_style": "Эмоциональный"
        }
        # Мокируем analyze, чтобы он возвращал словарь, а не строку
        self.mock_detector_agent.analyze.return_value = mock_analysis_data

        with patch('orchestrator.orchestrator.RUSSIAN_TO_INTERNAL_BIAS_MAP', {'Катастрофизация': 'catastrophizing'}):
             # Вызываем внутренний метод напрямую, минуя поток
            self.orchestrator._run_analysis_in_background("Все ужасно, я провалюсь, это конец света")

        self.mock_memory.save_cognitive_pattern.assert_called_once_with(
            pattern_name='catastrophizing', confidence=90, context="Контекст..."
        )
        self.mock_memory.save_psycholinguistic_features.assert_called_once_with(
            emotional_tone="Тревога", communication_style="Эмоциональный"
        )

    def test_extract_name_positive(self):
        self.assertEqual(self.orchestrator._extract_name("Меня зовут Алексей"), "Алексей")

    def test_get_greeting_new_user(self):
        self.mock_memory.get_user_name.return_value = None
        self.mock_memory.get_last_session_summary.return_value = None
        self.assertIn("как я могу к вам обращаться?", self.orchestrator.get_greeting())

    def test_get_greeting_returning_user_with_summary(self):
        self.mock_memory.get_user_name.return_value = "Анна"
        self.mock_memory.get_last_session_summary.return_value = "обсуждали котиков"
        self.assertIn("Анна, рад вас снова видеть!", self.orchestrator.get_greeting())
        self.assertIn("обсуждали котиков", self.orchestrator.get_greeting())

if __name__ == '__main__':
    unittest.main()
