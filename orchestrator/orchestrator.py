from enum import Enum, auto
import uuid
import json
from agents.task_agent import TaskAgent
from agents.detector_agent import DetectorAgent
from agents.methodology_agent import MethodologyAgent
from orchestrator.dynamic_memory import DynamicMemory
from orchestrator.action_library import ActionLibrary
from database.db_connector import get_chroma_collection

class AgentMode(Enum):
    COPILOT = "copilot"
    PARTNER = "partner"

class PartnerState(Enum):
    IDLE = auto()
    AWAITING_PROBLEM = auto()
    DECONSTRUCTION_COMPLETE = auto()
    HYPOTHESIS_COMPLETE = auto()

class Orchestrator:
    def __init__(self, user_id_stub="default_user"):
        self.user_id_stub = user_id_stub
        self.mode = AgentMode.COPILOT

        self.task_agent = TaskAgent()
        self.detector_agent = DetectorAgent()
        self.memory = DynamicMemory(user_id_stub)
        self.action_library = ActionLibrary()

        self.history_collection = get_chroma_collection("dialogue_history")
        self.partnership_proposed = False

        # Состояние для режима "Партнер"
        self.partner_state = PartnerState.IDLE
        self.last_partner_result = None

        print(f"Оркестратор инициализирован для пользователя {self.user_id_stub}.")

    def process_input(self, text: str) -> str:
        # Контур Б: Мета-анализ
        analysis_result = self.detector_agent.analyze(text)
        detected_patterns = json.loads(analysis_result)
        for pattern in detected_patterns:
            self.memory.save_cognitive_pattern(
                pattern_name=pattern['bias'],
                confidence=pattern['confidence'],
                context=pattern['context']
            )

        # Основная логика
        if self.mode == AgentMode.COPILOT:
            response = self.handle_copilot_mode(text, detected_patterns)
        elif self.mode == AgentMode.PARTNER:
            response = self.handle_partner_mode(text)
        else:
            response = "Ошибка: неизвестный режим работы."

        self.save_to_history(user_input=text, agent_output=response)
        return response

    def handle_copilot_mode(self, text, detected_patterns):
        response = self.task_agent.process(text)
        if detected_patterns and not self.partnership_proposed and detected_patterns[0]['confidence'] > 80:
            response += "\n\nКстати, я заметил некоторые особенности в нашем диалоге. Хотите переключиться в режим 'Партнер', чтобы мы могли поработать над этим вместе? (введите '/partner')"
            self.partnership_proposed = True
        return response

    def handle_partner_mode(self, text: str) -> str:
        if self.partner_state == PartnerState.IDLE:
            self.partner_state = PartnerState.AWAITING_PROBLEM
            return "Режим 'Партнер' активирован. Пожалуйста, опишите проблему, над которой вы хотите поработать."

        elif self.partner_state == PartnerState.AWAITING_PROBLEM:
            response = "Отлично, начинаем работу. Запускаю модуль 'Деконструкция'...\n\n"
            deconstruction_result = self.action_library.run_deconstruction(text)
            self.last_partner_result = deconstruction_result
            self.partner_state = PartnerState.DECONSTRUCTION_COMPLETE
            response += deconstruction_result
            response += "\n\n--- \nМы завершили деконструкцию. Хотите продолжить и сгенерировать гипотезы? (введите 'да' или любую другую фразу)"
            return response

        elif self.partner_state == PartnerState.DECONSTRUCTION_COMPLETE:
            response = "Приступаю к генерации гипотез на основе полученных фактов...\n\n"
            hypothesis_result = self.action_library.run_hypothesis_field(self.last_partner_result)
            self.last_partner_result = hypothesis_result
            self.partner_state = PartnerState.HYPOTHESIS_COMPLETE
            response += hypothesis_result
            response += "\n\n--- \nПоле гипотез создано. Следующий шаг — стресс-тестирование. (Эта функция будет добавлена позже). Чтобы начать новый разбор, введите '/reset'."
            return response

        elif self.partner_state == PartnerState.HYPOTHESIS_COMPLETE:
             return "Мы завершили генерацию гипотез. Следующие шаги (Стресс-тестинг, Синтез) будут реализованы в будущем. Чтобы начать новый анализ, введите '/reset'."


    def switch_mode(self, new_mode: AgentMode):
        self.mode = new_mode
        self.partnership_proposed = False
        if new_mode == AgentMode.PARTNER:
            self.partner_state = PartnerState.IDLE
        print(f"Режим изменен на: {self.mode.value}")

    def reset_partner_session(self):
        """Сбрасывает состояние сессии в режиме 'Партнер'."""
        self.partner_state = PartnerState.IDLE
        self.last_partner_result = None
        print("Сессия 'Партнер' сброшена.")

    def save_to_history(self, user_input: str, agent_output: str):
        if self.history_collection is None: return
        try:
            self.history_collection.add(
                ids=[str(uuid.uuid4())],
                documents=[f"Пользователь: {user_input}\nАгент: {agent_output}"],
                metadatas=[{"user_id": self.user_id_stub, "mode": self.mode.value}]
            )
        except Exception as e:
            print(f"Ошибка при сохранении диалога в ChromaDB: {e}")
