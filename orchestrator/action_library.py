from agents.methodology_agent import MethodologyAgent

class ActionLibrary:
    """
    Репозиторий программных функций, которые Оркестратор может вызывать
    для реализации "маршрутов" эксперта.
    """
    def __init__(self):
        self.methodology_agent = MethodologyAgent()
        print("Библиотека действий инициализирована.")

    def run_deconstruction(self, user_narrative: str) -> str:
        """
        Запускает модуль "Деконструкция".
        """
        system_prompt = (
            "Ты — AI-методолог. Твоя задача — помочь пользователю деконструировать его проблему. "
            "Преобразуй его нарратив в структурированную 'карту фактов'. "
            "Отделяй объективные факты от субъективных оценок. "
            "Представь результат в виде списка ключевых фактов."
        )
        print("Запуск модуля 'Деконструкция'...")
        return self.methodology_agent.execute(system_prompt, user_narrative)

    def run_hypothesis_field(self, fact_map: str) -> str:
        """
        Запускает модуль "Поле Гипотез".
        Генерирует 3-4 взаимоисключающие гипотезы на основе карты фактов.
        """
        system_prompt = (
            "Ты — AI-стратег. Твоя задача — сгенерировать поле гипотез на основе представленных фактов. "
            "Создай 3-4 взаимоисключающие гипотезы, которые объясняют эти факты. "
            "Гипотезы должны быть разнообразными: одна очевидная, одна инвертированная (противоположная), одна творческая (аналогия)."
        )
        print("Запуск модуля 'Поле Гипотез'...")
        user_prompt = f"Вот карта фактов, которую мы составили:\n\n{fact_map}\n\nПожалуйста, предложи несколько гипотез для их объяснения."
        return self.methodology_agent.execute(system_prompt, user_prompt)

if __name__ == '__main__':
    # Пример использования
    library = ActionLibrary()
    narrative = "Наш новый стартап не взлетает. Пользователи не приходят, а инвесторы начинают сомневаться."

    # Шаг 1: Деконструкция
    deconstruction_result = library.run_deconstruction(narrative)
    print(f"\n--- Результат деконструкции ---\n{deconstruction_result}")

    # Шаг 2: Поле Гипотез
    hypothesis_result = library.run_hypothesis_field(deconstruction_result)
    print(f"\n--- Результат Поля Гипотез ---\n{hypothesis_result}")
