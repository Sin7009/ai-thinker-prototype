from agents.methodology_agent import MethodologyAgent

class ActionLibrary:
    """
    Репозиторий конкретных мыслительных техник, которые Оркестратор может предлагать пользователю.
    Каждая техника — это отдельный метод, который вызывает MethodologyAgent с особым системным промптом.
    """
    def __init__(self, methodology_agent: MethodologyAgent):
        self.methodology_agent = methodology_agent
        print("Библиотека действий инициализирована.")

    def run_rubber_duck_debugging(self, problem_description: str) -> str:
        """
        Проводит пользователя через "Метод утёнка".
        """
        system_prompt = (
            "Твоя роль — 'резиновый утёнок'. Я ничего не знаю о твоей проблеме. "
            "Твоя задача — слушать и задавать простые уточняющие вопросы. "
            "Не предлагай решений. Пусть пользователь сам найдет ответ, объясняя проблему."
            "\n\n**Важное правило безопасности:** Если пользователь хочет остановиться, верни `[STOP_TECHNIQUE]`."
        )
        print("Запуск техники 'Метод утёнка'...")
        return self.methodology_agent.execute(system_prompt, problem_description)

    def run_five_whys(self, initial_problem: str) -> str:
        """
        Проводит пользователя через технику "Пять почему" для поиска корневой причины.
        """
        system_prompt = (
            "Твоя роль — коуч, использующий технику 'Пять почему'. "
            "Задавай вопрос 'Почему?' к каждому утверждению пользователя, пока не найдешь корневую причину. "
            "В конце подрезюмируй."
            "\n\n**Важное правило безопасности:** Для выхода верни `[STOP_TECHNIQUE]`."
        )
        print("Запуск техники 'Пять почему'...")
        return self.methodology_agent.execute(system_prompt.format(initial_problem=initial_problem), initial_problem)

    def run_constrained_brainstorming(self, topic: str) -> str:
        """
        Проводит сессию мозгового штурма с искусственными ограничениями для стимулирования креативности.
        """
        system_prompt = (
            "Твоя роль — фасилитатор мозгового штурма. Введи неожиданное ограничение (бюджет 100р, время 1 час и т.д.) "
            "и попроси накидать идеи."
            "\n\n**Важное правило безопасности:** Для выхода верни `[STOP_TECHNIQUE]`."
        )
        print("Запуск техники 'Мозговой штурм с ограничениями'...")
        return self.methodology_agent.execute(system_prompt.format(topic=topic), topic)
