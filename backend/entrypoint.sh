#!/usr/bin/env bash
set -e

# применяем миграции
alembic upgrade head

# запускаем API
# Вариант 1: через uvicorn напрямую
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

# Вариант 2 (если хочешь использовать твой модульный раннер и hot-reload):
# exec python -m app.__run__
