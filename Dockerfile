# Используем slim-версию для уменьшения размера образа
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
# Это делается отдельным слоем для кэширования Docker

COPY requirements.txt .

# 1. Сначала ставим легкий CPU-PyTorch отдельно
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Остальные зависимости
# pip увидит, что torch уже стоит, и не будет качать гигабайтную версию
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код проекта
COPY . .

# Переменные окружения (можно переопределить в docker-compose)
ENV PYTHONUNBUFFERED=1

# Запуск бота по умолчанию (не main.py, а telegram_bot.py)
CMD ["python", "telegram_bot.py"]
