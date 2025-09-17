# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Библиотеки для opencv-headless и др.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Сначала только зависимости (лучше кэшируется)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем — весь код
COPY . .

# На входе в контейнер выполним миграции и запустим uvicorn
# (см. обновлённый entrypoint.sh)
ENTRYPOINT ["./entrypoint.sh"]
