"""Вебхук Small Talk и фоновая обработка входящих сообщений."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from openai import AsyncOpenAI, OpenAIError

from init_db import database, init_db

logger = logging.getLogger("uvicorn.error")
SYSTEM_PROMPT = (
    "Ты — дружелюбный ИИ-тренер по имени Small Talk. "
    "Твой ответ должен быть строго в формате: REPLY: "
    "[короткая реплика на английском] и FEEDBACK: "
    "[мягкое исправление ошибки на русском, если есть]."
)


@dataclass(frozen=True)
class IncomingMessage:
    source: str
    chat_id: str
    message_id: str
    user_id: str
    text: str


def object_field(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def identifier(value: Any) -> str:
    if type(value) in (int, str) and str(value).strip():
        return str(value)
    return ""


def parse_message(payload: dict) -> IncomingMessage | None:
    message = object_field(payload.get("message"))
    if payload.get("update_type") == "message_created":
        source = "max"
        body = object_field(message.get("body"))
        recipient = object_field(message.get("recipient"))
        # MAX присваивает сообщению mid, например mid.000…
        chat_id = identifier(recipient.get("chat_id"))
        message_id = identifier(body.get("mid"))
        user_id = identifier(object_field(message.get("sender")).get("user_id"))
        text = body.get("text")
    elif "message" in payload and "update_type" not in payload:
        source = "telegram"
        # Telegram нумерует сообщения в пределах чата.
        chat_id = identifier(object_field(message.get("chat")).get("id"))
        message_id = identifier(message.get("message_id"))
        user_id = identifier(object_field(message.get("from")).get("id"))
        text = message.get("text")
    else:
        # Плоский формат удобен для локальной проверки через curl.
        source = "local"
        chat_id = identifier(payload.get("chat_id"))
        message_id = identifier(payload.get("message_id"))
        user_id = identifier(payload.get("user_id"))
        text = payload.get("text")

    if not message_id or not user_id or not isinstance(text, str) or not text.strip():
        return None
    if source == "telegram" and not chat_id:
        return None
    return IncomingMessage(source, chat_id or user_id, message_id, user_id, text)


async def process_message(client: AsyncOpenAI, lock: asyncio.Lock, job_id: int):
    # Один процесс Uvicorn сериализует работу с диалогами. SQLite-транзакции
    # не остаются открытыми во время сетевого запроса.
    async with lock:
        with database() as connection:
            job = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND completed = 0", (job_id,)
            ).fetchone()
            if job is None:
                return
            history = connection.execute(
                """SELECT role, content FROM history
                   WHERE source = ? AND user_id = ?
                   ORDER BY id DESC LIMIT 5""",
                (job["source"], job["user_id"]),
            ).fetchall()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(dict(row) for row in reversed(history)),
            {"role": "user", "content": job["text"]},
        ]
        try:
            result = await client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=messages,
            )
        except OpenAIError as error:
            # Задача остаётся в SQLite для повторной попытки при перезапуске.
            logger.error("Groq failed for job %s (%s)", job_id, type(error).__name__)
            return

        reply = result.choices[0].message.content if result.choices else None
        if not reply or not reply.strip():
            logger.error("Groq returned no text for job %s", job_id)
            return

        # История и признак завершения фиксируются одной транзакцией.
        with database() as connection:
            connection.executemany(
                """INSERT INTO history (source, user_id, role, content)
                   VALUES (?, ?, ?, ?)""",
                [
                    (job["source"], job["user_id"], "user", job["text"]),
                    (job["source"], job["user_id"], "assistant", reply),
                ],
            )
            connection.execute("UPDATE jobs SET completed = 1 WHERE id = ?", (job_id,))
        logger.info("Small Talk job %s: %s", job_id, reply)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Укажите GROQ_API_KEY в .env или окружении.")
    async with AsyncOpenAI(
        api_key=key, base_url="https://api.groq.com/openai/v1", timeout=20.0
    ) as client:
        app.state.groq = client
        app.state.processing_lock = asyncio.Lock()

        async def resume_pending():
            with database() as connection:
                pending = connection.execute(
                    "SELECT id FROM jobs WHERE completed = 0 ORDER BY id"
                ).fetchall()
            for row in pending:
                await process_message(client, app.state.processing_lock, row["id"])

        recovery = asyncio.create_task(resume_pending())
        try:
            yield
        finally:
            recovery.cancel()
            await asyncio.gather(recovery, return_exceptions=True)


app = FastAPI(title="Small Talk", lifespan=lifespan)


@app.post("/webhook")
def webhook(payload: dict, background_tasks: BackgroundTasks):
    incoming = parse_message(payload)
    if incoming is None:
        if "message_id" in payload or "user_id" in payload or "text" in payload:
            raise HTTPException(status_code=422, detail="Invalid message payload")
        # Мессенджеры присылают и события без текстовых сообщений.
        return {"ok": True}

    # Уникальное ограничение защищает и от одновременно пришедших дублей.
    # Коммит выполняется прежде, чем сервер отправит 200 и запустит задачу.
    with database() as connection:
        cursor = connection.execute(
            """INSERT INTO jobs (source, chat_id, message_id, user_id, text)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (source, chat_id, message_id) DO NOTHING""",
            (incoming.source, incoming.chat_id, incoming.message_id,
             incoming.user_id, incoming.text),
        )
        job_id = cursor.lastrowid if cursor.rowcount else None
    if job_id is not None:
        background_tasks.add_task(
            process_message, app.state.groq, app.state.processing_lock, job_id
        )
    return {"ok": True}
