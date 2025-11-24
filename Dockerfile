# Используем slim-версию для уменьшения размера образа
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# 1. Сначала ставим легкий CPU-PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# --- НОВЫЙ БЛОК: ЗАПЕКАНИЕ МОДЕЛЕЙ ---
# Копируем скрипт пре-загрузки
COPY preload_models.py .
# Запускаем его. Это скачает модели и сохранит их в слое Docker-образа.
RUN python preload_models.py
# Удаляем скрипт, он больше не нужен
RUN rm preload_models.py
# -------------------------------------

# Копируем исходный код проекта
COPY . .

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "telegram_bot.py"]
