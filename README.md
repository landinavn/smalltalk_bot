# Small Talk

Small Talk — единая платформа для практики разговорного английского с AI-преподавателем.

Продукт предназначен для работы через Telegram и MAX. Это один сервис с общей бизнес-логикой, профилями пользователей, прогрессом и подписками, а не два независимых приложения.

## Структура

- `main.py` — существующий Python backend.
- `init_db.py` — инициализация базы данных.
- `requirements.txt` — Python-зависимости.
- `frontend/` — React + TypeScript + Vite + Three.js.
- `docs/` — техническая документация, архитектура и план разработки.

## Технологии

Backend: Python, FastAPI, SQLite, OpenAI-compatible API.

Frontend: React, TypeScript, Vite, Three.js.

## Документация

Перед любыми изменениями кода изучите:

1. `docs/PROJECT_AUDIT.md`
2. `docs/ARCHITECTURE_PLAN.md`
3. `docs/IMPLEMENTATION_ROADMAP.md`
4. `docs/CODEX_INSTRUCTIONS.md`

## Важно

Работающий backend является production-сервисом. Изменения в нём, перезапуск, миграции базы данных и изменение инфраструктуры допускаются только после отдельного согласования.

Проект `Bot-Max` не относится к Small Talk. Запрещено обращаться к его файлам, настройкам, токенам и службам.
