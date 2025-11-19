import unittest
import subprocess
import sys

def install_dependencies():
    packages = [
        "langchain",
        "langchain-community",
        "langchain-gigachat",
        "chromadb",
        "sentence-transformers",
        "sqlalchemy"
    ]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_dependencies()
from unittest.mock import MagicMock, patch
from orchestrator.orchestrator import Orchestrator, AgentMode
from orchestrator.action_library import ActionLibrary
from agents.methodology_agent import MethodologyAgent
from database.db_connector import recreate_tables
import json
import time
import os

# Пересоздаем таблицы перед запуском всех тестов
recreate_tables()

@patch('orchestrator.orchestrator.DynamicMemory')
@patch('orchestrator.orchestrator.MethodologyAgent')
@patch('orchestrator.orchestrator.DetectorAgent')
@patch('orchestrator.orchestrator.TaskAgent')
class TestOrchestrator(unittest.TestCase):

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_full_thinking_cycle(self, mock_develop_strategy, MockTaskAgent, MockDetectorAgent, MockMethodologyAgent, MockDynamicMemory):
        """
        Тест: Полный "мыслительный цикл" от ключевой фразы до финального ответа.
        """
        orchestrator = Orchestrator(user_id_stub="test_user")
        # Настраиваем возвращаемые значения моков
        orchestrator.task_agent.process.return_value = 'run_five_whys'
        orchestrator.action_library.methodology_agent.invoke.return_value = 'Это финальный ответ от техники пяти почему.'
        orchestrator.detector_agent.analyze.return_value = "{}"

        # Запускаем процесс
        final_response = orchestrator.process_input("Давай подумаем над моей проблемой")

        # Проверяем результаты
        self.assertEqual(orchestrator.mode, AgentMode.PARTNER)
        orchestrator.task_agent.process.assert_called_once()
        orchestrator.action_library.methodology_agent.invoke.assert_called_once()
        self.assertEqual(final_response, 'Это финальный ответ от техники пяти почему.')

        # Проверяем детали вызовов
        diagnosis_call = orchestrator.task_agent.process.call_args
        self.assertIn("'run_five_whys'", diagnosis_call.kwargs['context_memory'])

        technique_call = orchestrator.action_library.methodology_agent.invoke.call_args
        self.assertIn("Твоя роль — коуч, использующий технику 'Пять почему'", str(technique_call.args[0]))

    @patch('threading.Thread.start')
    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_analysis_data_is_parsed_and_saved(self, mock_develop_strategy, mock_thread_start, MockTaskAgent, MockDetectorAgent, MockMethodologyAgent, MockDynamicMemory):
        """Тест: Оркестратор парсит комплексный JSON и сохраняет все данные."""
        orchestrator = Orchestrator(user_id_stub="test_user")
        mock_analysis_data = {
            "cognitive_biases": [{"name": "Катастрофизация", "confidence": 90, "context": "Контекст..."}],
            "emotional_tone": "Тревога",
            "communication_style": "Эмоциональный"
        }
        # Мокируем analyze, чтобы он возвращал словарь, а не строку
        orchestrator.detector_agent.analyze.return_value = mock_analysis_data

        with patch('orchestrator.orchestrator.RUSSIAN_TO_INTERNAL_BIAS_MAP', {'Катастрофизация': 'catastrophizing'}):
             # Вызываем внутренний метод напрямую, минуя поток
            orchestrator._run_analysis_in_background("Все ужасно, я провалюсь, это конец света")

        orchestrator.memory.save_cognitive_pattern.assert_called_once_with(
            pattern_name='catastrophizing', confidence=90, context="Контекст..."
        )
        orchestrator.memory.save_psycholinguistic_features.assert_called_once_with(
            emotional_tone="Тревога", communication_style="Эмоциональный"
        )

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_extract_name_positive(self, mock_develop_strategy, MockTaskAgent, MockDetectorAgent, MockMethodologyAgent, MockDynamicMemory):
        orchestrator = Orchestrator(user_id_stub="test_user")
        self.assertEqual(orchestrator._extract_name("Меня зовут Алексей"), "Алексей")

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_get_greeting_new_user(self, mock_develop_strategy, MockTaskAgent, MockDetectorAgent, MockMethodologyAgent, MockDynamicMemory):
        orchestrator = Orchestrator(user_id_stub="test_user")
        orchestrator.memory.get_user_name.return_value = None
        orchestrator.memory.get_last_session_summary.return_value = None
        self.assertIn("как я могу к вам обращаться?", orchestrator.get_greeting())

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_get_greeting_returning_user_with_summary(self, mock_develop_strategy, MockTaskAgent, MockDetectorAgent, MockMethodologyAgent, MockDynamicMemory):
        orchestrator = Orchestrator(user_id_stub="test_user")
        orchestrator.memory.get_user_name.return_value = "Анна"
        orchestrator.memory.get_last_session_summary.return_value = "обсуждали котиков"
        self.assertIn("Анна, рад вас снова видеть!", orchestrator.get_greeting())
        self.assertIn("обсуждали котиков", orchestrator.get_greeting())

if __name__ == '__main__':
    unittest.main()
