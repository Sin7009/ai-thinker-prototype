import os
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from orchestrator.orchestrator import Orchestrator, AgentMode

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Глобальный словарь сессий: {telegram_user_id: OrchestratorInstance}
# В продакшене лучше использовать Redis, но для прототипа словарь в памяти подойдет.
user_sessions = {}

def get_orchestrator(user_id: int) -> Orchestrator:
    """Ленивая инициализация Оркестратора для конкретного юзера."""
    if user_id not in user_sessions:
        # Используем ID телеграма как уникальный stub
        print(f"Создаю новую сессию для user_id: {user_id}")
        user_sessions[user_id] = Orchestrator(user_id_stub=str(user_id))
    return user_sessions[user_id]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orc = get_orchestrator(update.effective_user.id)
    # Генерируем приветствие, используя логику Оркестратора
    greeting = orc.get_greeting()
    await update.message.reply_text(greeting)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Доступные команды:\n"
        "/partner — Режим методолога (5 почему, Утенок)\n"
        "/copilot — Режим прямого ассистента\n"
        "/memory — Что я о вас знаю\n"
        "/reset — Сброс контекста"
    )
    await update.message.reply_text(help_text)

async def switch_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orc = get_orchestrator(update.effective_user.id)
    command = update.message.text.lower()

    if '/partner' in command:
        orc.switch_mode(AgentMode.PARTNER)
        msg = "Режим: ПАРТНЕР. Я буду задавать вопросы и использовать техники мышления."
    elif '/copilot' in command:
        orc.switch_mode(AgentMode.COPILOT)
        msg = "Режим: КОПИЛОТ. Отвечаю прямо и по делу."

    await update.message.reply_text(msg)

async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orc = get_orchestrator(update.effective_user.id)
    # Используем метод получения саммари профиля
    summary = orc.memory.get_user_profile_summary()
    await update.message.reply_text(f"🧠 Моя память о вас:\n\n{summary}")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orc = get_orchestrator(update.effective_user.id)
    orc.reset_all_memory()
    await update.message.reply_text("🗑 Оперативная память очищена. Начинаем с чистого листа.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    orc = get_orchestrator(user_id)

    # Важно: process_input — синхронная функция (блокирующая), а бот асинхронный.
    # Чтобы бот не "замирал" для других юзеров во время генерации ответа,
    # запускаем обработку в отдельном потоке.
    response = await asyncio.to_thread(orc.process_input, text)

    await update.message.reply_text(response)

if __name__ == '__main__':
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("Переменная окружения TELEGRAM_TOKEN не установлена!")

    application = ApplicationBuilder().token(token).build()

    # Регистрация хендлеров
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler(['partner', 'copilot'], switch_mode))
    application.add_handler(CommandHandler('memory', show_memory))
    application.add_handler(CommandHandler('reset', reset))

    # Обработка текста
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Бот запущен...")
    application.run_polling()
