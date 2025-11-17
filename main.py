from orchestrator.orchestrator import Orchestrator, AgentMode

def main():
    print("Добро пожаловать в AI-Мыслитель!")
    print("Команды: /partner, /copilot, /reset, /memory, /exit")

    orchestrator = Orchestrator(user_id_stub="default_user")

    try:
        while True:
            user_input = input("Вы: ")
            orchestrator.last_user_input = user_input

            # 💬 Обработка команд
            if user_input.lower() == '/exit':
                orchestrator.end_session()
                print("Агент: До свидания! Был рад помочь.")
                break

            if user_input.lower() == '/partner':
                orchestrator.switch_mode(AgentMode.PARTNER)
                print("Агент: Режим переключён на 'Партнёр'. Опишите, над чем хотите поработать.")
                continue

            if user_input.lower() == '/copilot':
                orchestrator.switch_mode(AgentMode.COPILOT)
                print("Агент: Режим переключён на 'Копилот'.")
                continue

            if user_input.lower() == '/reset':
                orchestrator.reset_all_memory()
                print("Агент: Вся память сброшена. Диалог начат заново.")
                continue

            if user_input.lower() == '/memory':
                summary = orchestrator.memory.get_user_profile_summary()
                print(f"Агент: {summary}")
                continue

            # 🔔 Автонапоминание — теперь здесь, после первого ввода
            if not hasattr(orchestrator, '_greeted') and user_input.strip():
                name = orchestrator.memory.get_user_name()
                last_summary = orchestrator.memory.get_last_session_summary()

                if name:
                    print(f"Агент: Привет, {name}! Рад снова тебя видеть.")
                    if last_summary:
                        print(f"Агент: В прошлый раз мы говорили о:\n{last_summary}")
                        print("Агент: Как продвигается эта тема? Нужна помощь с анализом или решением?")
                    else:
                        print("Агент: Чем займёмся сегодня?")
                else:
                    print("Агент: Добрый день! Я — ваш AI-мышлитель.")
                    print("Агент: Готов помочь с решением задач, анализом мышления или просто поговорить.")
                    print("Агент: Как я могу к вам обращаться?")

                orchestrator._greeted = True  # ← Защита от повторного приветствия
            # 🔤 Попытка извлечь имя — только если его ещё нет
            if not orchestrator.memory.get_user_name():
                name = orchestrator._extract_name(user_input)
                if name:
                    orchestrator.memory.save_user_name(name)
                    print(f"Агент: Отлично, {name}! Теперь я знаю, как к вам обращаться.")

            # 🚀 Обработка ввода
            response = orchestrator.process_input(user_input)
            print(f"Агент: {response}")

    except KeyboardInterrupt:
        orchestrator.end_session()
        print("\nАгент: До свидания!")

if __name__ == "__main__":
    main()
