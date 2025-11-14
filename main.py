from orchestrator.orchestrator import Orchestrator, AgentMode

def main():
    """
    Основная функция для запуска консольного интерфейса AI-агента.
    """
    print("Добро пожаловать в AI-Мыслитель!")
    print("Команды: /partner, /copilot, /reset, /exit")

    orchestrator = Orchestrator(user_id_stub="default_user")

    while True:
        try:
            user_input = input("Вы: ")

            if user_input.lower() in ['exit', 'quit']:
                print("До свидания!")
                break

            if user_input.lower() == '/partner':
                orchestrator.switch_mode(AgentMode.PARTNER)
                # Сразу выводим первое сообщение режима "Партнер"
                response = orchestrator.handle_partner_mode(user_input)
                print(f"Агент: {response}")
                continue

            if user_input.lower() == '/copilot':
                orchestrator.switch_mode(AgentMode.COPILOT)
                print("Агент: Режим переключен на 'Копилот'.")
                continue

            if user_input.lower() == '/reset':
                if orchestrator.mode == AgentMode.PARTNER:
                    orchestrator.reset_partner_session()
                    print("Агент: Сессия 'Партнер' сброшена.")
                else:
                    print("Агент: Команда /reset работает только в режиме 'Партнер'.")
                continue

            response = orchestrator.process_input(user_input)
            print(f"Агент: {response}")

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
