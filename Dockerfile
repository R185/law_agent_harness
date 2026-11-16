FROM python:3.12-slim

# Установка системных утилит для healthcheck и базовой работы
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Оптимизация Python для работы в контейнере
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8765 \
    HOST=0.0.0.0

# Установка зависимостей с кэшированием слоёв
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта и базы знаний
COPY . /app/

# Порт Web UI и API сервиса
EXPOSE 8765

# Проверка работоспособности сервиса
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:' + str(__import__('os').getenv('PORT', 8765)) + '/api/health')"

# Запуск Web UI и API сервера по умолчанию
CMD ["uvicorn", "ui.server:app", "--host", "0.0.0.0", "--port", "8765", "--ws", "none"]
