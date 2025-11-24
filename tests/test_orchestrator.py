import unittest
from unittest.mock import MagicMock, patch
from orchestrator.orchestrator import Orchestrator, AgentMode
import json

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Патчим память, чтобы не трогать реальную базу данных
        with patch('orchestrator.orchestrator.DynamicMemory') as mock_dynamic_memory:
            self.orchestrator = Orchestrator(user_id_stub="test_user")
            self.orchestrator.memory = mock_dynamic_memory.return_value

        # Создаем моки для всех агентов
        self.mock_task_agent = MagicMock()
        self.mock_methodology_agent = MagicMock()
        self.mock_detector_agent = MagicMock()

        # Подменяем реальных агентов на моки
        self.orchestrator.task_agent = self.mock_task_agent
        self.orchestrator.detector_agent = self.mock_detector_agent
        self.orchestrator.methodology_agent = self.mock_methodology_agent

        # Обновляем ActionLibrary, чтобы она использовала наш мок методологического агента
        self.orchestrator.action_library.methodology_agent = self.mock_methodology_agent

        # Базовый ответ от TaskAgent (чтобы не падал на json.loads в некоторых местах, если нужно)
        self.mock_task_agent.process.return_value = '[]'

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_full_thinking_cycle(self, mock_develop_strategy):
        """
        Тест: Полный "мыслительный цикл" от ключевой фразы до финального ответа.
        Проверяет, что вызывается execute() вместо invoke().
        """
        # Мокируем ответ диагностического агента (TaskAgent), чтобы он выбрал технику
        self.mock_task_agent.process.return_value = 'run_five_whys'

        # Мокируем ответ методологического агента (через execute)
        self.mock_methodology_agent.execute.return_value = 'Ответ техники 5 почему.'

        # Мокируем ответ детектора (анализ текста) - пустой JSON
        self.mock_detector_agent.analyze.return_value = {}

        # Запускаем обработку ввода
        final_response = self.orchestrator.process_input("Давай подумаем")

        # Проверяем переключение режима
        self.assertEqual(self.orchestrator.mode, AgentMode.PARTNER)

        # Проверяем, что был вызван execute (а не invoke)
        self.mock_methodology_agent.execute.assert_called_once()

        # Проверяем финальный ответ
        self.assertEqual(final_response, 'Ответ техники 5 почему.')

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_extract_name_positive(self, mock_develop_strategy):
        """Тест извлечения имени (сохранен из старого набора тестов)."""
        self.assertEqual(self.orchestrator._extract_name("Меня зовут Алексей"), "Алексей")

    @patch('orchestrator.orchestrator.Orchestrator._develop_strategy')
    def test_analysis_data_is_parsed_and_saved(self, mock_develop_strategy):
        """
        Тест: Оркестратор парсит комплексный JSON от детектора и сохраняет данные.
        Адаптирован под новую структуру setUp.
        """
        mock_analysis_data = {
            "cognitive_biases": [{"name": "Катастрофизация", "confidence": 90, "context": "Контекст..."}],
            "emotional_tone": "Тревога",
            "communication_style": "Эмоциональный"
        }
        # Мокируем analyze
        self.mock_detector_agent.analyze.return_value = mock_analysis_data

        # Патчим маппинг, чтобы тест не зависел от реального файла маппинга
        with patch('orchestrator.orchestrator.RUSSIAN_TO_INTERNAL_BIAS_MAP', {'Катастрофизация': 'catastrophizing'}):
             # Вызываем внутренний метод напрямую (синхронно для теста)
            self.orchestrator._run_analysis_in_background("Все ужасно, я провалюсь")

        # Проверяем вызовы методов сохранения памяти
        self.orchestrator.memory.save_cognitive_pattern.assert_called_once_with(
            pattern_name='catastrophizing', confidence=90, context="Контекст..."
        )
        self.orchestrator.memory.save_psycholinguistic_features.assert_called_once_with(
            emotional_tone="Тревога", communication_style="Эмоциональный"
        )

if __name__ == '__main__':
    unittest.main()
