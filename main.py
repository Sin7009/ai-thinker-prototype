from orchestrator.orchestrator import Orchestrator, AgentMode

def main():
    print("Добро пожаловать в AI-Мыслитель!")
    print("Команды: /partner, /copilot, /reset, /memory, /exit")

    orchestrator = Orchestrator(user_id_stub="default_user")

    try:
        # 1. Сначала бот приветствует пользователя
        greeting = orchestrator.get_greeting()
        print(f"Агент: {greeting}")

        # 2. Затем начинается цикл диалога
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

            # 🔤 Попытка извлечь имя — только если его ещё нет
            if not orchestrator.memory.get_user_name():
                name = orchestrator._extract_name(user_input)
                if name:
                    orchestrator.memory.save_user_name(name)
                    print(f"Агент: Отлично, {name}! Теперь я знаю, как к вам обращаться.")
                    # После того как имя получено, можно сразу перейти к следующему вводу,
                    # чтобы не обрабатывать имя как обычный запрос
                    continue

            # 🚀 Обработка ввода
            response = orchestrator.process_input(user_input)
            print(f"Агент: {response}")

    except KeyboardInterrupt:
        orchestrator.end_session()
        print("\nАгент: До свидания!")

if __name__ == "__main__":
    main()
