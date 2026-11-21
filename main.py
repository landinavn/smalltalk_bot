"""Рабочая версия Small Talk: MAX webhook, Groq, голосовые функции."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from openai import AsyncOpenAI, OpenAIError

from init_db import database, init_db

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

logger = logging.getLogger("uvicorn.error")

MAX_API = "https://platform-api2.max.ru"
MAX_CA = "/etc/ssl/certs/ca-certificates.crt"
GROQ_PROXY = "http://94.249.177.108:3128"

SYSTEM_PROMPT = (
    "Ты — дружелюбный ИИ-тренер по имени Small Talk. "
    "Помогай пользователю практиковать английский в живом диалоге. "
    "Твой ответ должен быть строго в формате:\n"
    "REPLY: короткая реплика на английском.\n"
    "FEEDBACK: мягкое исправление ошибки на русском, если ошибка есть. "
    "Если ошибок нет, кратко похвали пользователя."
)

WELCOME_TEXT = (
    "Привет! Я Small Talk — твой ИИ-собеседник для практики английского. 👋\n\n"
    "Давай преодолевать языковой барьер через живое общение!\n\n"
    "Мы можем:\n"
    "🗣 Поболтать на свободную тему\n"
    "☕ Разыграть ситуацию в кафе, аэропорту или отеле\n"
    "💼 Подготовиться к собеседованию\n\n"
    "Напиши мне что-нибудь на английском — и начнём!"
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
        body = object_field(message.get("body"))
        recipient = object_field(message.get("recipient"))
        sender = object_field(message.get("sender"))

        source = "max"
        chat_id = identifier(recipient.get("chat_id"))
        user_id = identifier(sender.get("user_id"))
        message_id = identifier(body.get("mid"))
        text = body.get("text")

    elif "message" in payload and "update_type" not in payload:
        source = "telegram"
        chat_id = identifier(object_field(message.get("chat")).get("id"))
        message_id = identifier(message.get("message_id"))
        user_id = identifier(object_field(message.get("from")).get("id"))
        text = message.get("text")

    else:
        source = "local"
        chat_id = identifier(payload.get("chat_id"))
        message_id = identifier(payload.get("message_id"))
        user_id = identifier(payload.get("user_id"))
        text = payload.get("text")

    if not message_id or not user_id:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    if source == "telegram" and not chat_id:
        return None

    return IncomingMessage(source, chat_id, message_id, user_id, text.strip())


async def send_max_message(
    client: httpx.AsyncClient,
    token: str,
    text: str,
    user_id: str = "",
    chat_id: str = "",
):
    """Отправляет текст пользователю или в чат MAX."""
    if chat_id:
        params = {"chat_id": chat_id}
    elif user_id:
        params = {"user_id": user_id}
    else:
        raise ValueError("Не указан адресат сообщения MAX")

    response = await client.post(
        f"{MAX_API}/messages",
        params=params,
        headers={"Authorization": token},
        json={"text": text},
    )
    response.raise_for_status()
    return response


async def process_message(
    groq: AsyncOpenAI,
    max_client: httpx.AsyncClient,
    max_token: str,
    lock: asyncio.Lock,
    job_id: int,
):
    """Обрабатывает одну задачу. Старые задачи сами при запуске не запускаются."""
    async with lock:
        with database() as connection:
            job = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND completed = 0",
                (job_id,),
            ).fetchone()

            if job is None:
                return

            history = connection.execute(
                """SELECT role, content FROM history
                   WHERE source = ? AND user_id = ?
                   ORDER BY id DESC LIMIT 5""",
                (job["source"], job["user_id"]),
            ).fetchall()

            # Копируем данные до закрытия соединения.
            job_data = dict(job)
            history_data = [dict(row) for row in reversed(history)]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history_data,
            {"role": "user", "content": job_data["text"]},
        ]

        try:
            result = await groq.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                messages=messages,
            )
        except OpenAIError as error:
            logger.error(
                "Groq failed for job %s (%s)",
                job_id,
                type(error).__name__,
            )
            return
        except Exception as error:
            logger.error(
                "Generation failed for job %s (%s)",
                job_id,
                type(error).__name__,
            )
            return

        reply = (
            result.choices[0].message.content
            if result.choices
            else None
        )

        if not isinstance(reply, str) or not reply.strip():
            logger.error("Groq returned empty reply for job %s", job_id)
            return

        reply = reply.strip()

        # Для MAX сначала отправляем ответ. При ошибке задача остаётся
        # незавершённой, история не записывается.
        if job_data["source"] == "max":
            try:
                await send_max_message(
                    max_client,
                    max_token,
                    reply,
                    user_id=job_data["user_id"],
                    chat_id=job_data["chat_id"],
                )
            except Exception as error:
                logger.error(
                    "MAX send failed for job %s (%s)",
                    job_id,
                    type(error).__name__,
                )
                return

        elif job_data["source"] == "telegram":
            logger.error(
                "Telegram reply is not configured for job %s",
                job_id,
            )
            return

        # Сохраняем диалог и завершаем задачу одной транзакцией.
        with database() as connection:
            connection.executemany(
                """INSERT INTO history (source, user_id, role, content)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        job_data["source"],
                        job_data["user_id"],
                        "user",
                        job_data["text"],
                    ),
                    (
                        job_data["source"],
                        job_data["user_id"],
                        "assistant",
                        reply,
                    ),
                ],
            )
            connection.execute(
                "UPDATE jobs SET completed = 1 WHERE id = ?",
                (job_id,),
            )

        logger.info("Small Talk job %s completed", job_id)


async def generate_tts_en(
    client: AsyncOpenAI,
    text: str,
    output_path: str,
):
    """Озвучивает английский текст через Groq Orpheus."""
    try:
        response = await client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=text,
            response_format="wav",
        )
        await response.astream_to_file(output_path)
    except Exception as error:
        logger.error("TTS EN failed: %s", type(error).__name__)


async def generate_tts_ru(text: str, output_path: str):
    """Озвучивает русский текст через Edge TTS."""
    try:
        communicate = edge_tts.Communicate(
            text,
            voice="ru-RU-SvetlanaNeural",
        )
        await communicate.save(output_path)
    except Exception as error:
        logger.error("TTS RU failed: %s", type(error).__name__)


async def transcribe_audio(
    client: AsyncOpenAI,
    audio_file_path: str,
) -> str:
    """Распознаёт речь через Groq Whisper."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text",
            )
        return transcription.strip()
    except Exception as error:
        logger.error("STT failed: %s", type(error).__name__)
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    max_token = os.getenv("MAX_BOT_TOKEN", "").strip()

    if not groq_key:
        raise RuntimeError("Не задан GROQ_API_KEY в .env")
    if not max_token:
        raise RuntimeError("Не задан MAX_BOT_TOKEN в .env")

    # MAX работает напрямую с системным хранилищем сертификатов.
    async with httpx.AsyncClient(
        verify=MAX_CA,
        timeout=20.0,
    ) as max_client:
        async with AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=20.0,
            http_client=httpx.AsyncClient(
                proxy=GROQ_PROXY,
                timeout=20.0,
            ),
        ) as groq:
            app.state.groq = groq
            app.state.max_client = max_client
            app.state.max_token = max_token
            app.state.processing_lock = asyncio.Lock()

            # ВАЖНО: resume_pending отключён.
            # Старые незавершённые задачи не запускаются автоматически.
            logger.info("Small Talk started")
            try:
                yield
            finally:
                logger.info("Small Talk stopped")


app = FastAPI(title="Small Talk", lifespan=lifespan)


@app.post("/webhook")
async def webhook(
    payload: dict,
    background_tasks: BackgroundTasks,
):
    # Событие запуска бота в MAX не является обычным сообщением.
    if payload.get("update_type") == "bot_started":
        user = object_field(payload.get("user"))
        user_id = identifier(user.get("user_id"))
        chat_id = identifier(payload.get("chat_id"))

        if not user_id and not chat_id:
            return {"ok": True}

        try:
            await send_max_message(
                app.state.max_client,
                app.state.max_token,
                WELCOME_TEXT,
                user_id=user_id,
                chat_id=chat_id,
            )
        except Exception as error:
            logger.error(
                "MAX welcome failed (%s)",
                type(error).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Could not send welcome message",
            )

        return {"ok": True}

    incoming = parse_message(payload)

    if incoming is None:
        if (
            "message_id" in payload
            or "user_id" in payload
            or "text" in payload
        ):
            raise HTTPException(
                status_code=422,
                detail="Invalid message payload",
            )
        return {"ok": True}

    if incoming.text.lower() in ("/start", "начать"):
        if incoming.source == "max":
            try:
                await send_max_message(
                    app.state.max_client,
                    app.state.max_token,
                    WELCOME_TEXT,
                    user_id=incoming.user_id,
                    chat_id=incoming.chat_id,
                )
            except Exception as error:
                logger.error(
                    "MAX start welcome failed (%s)",
                    type(error).__name__,
                )
                raise HTTPException(
                    status_code=502,
                    detail="Could not send welcome message",
                )
        return {"ok": True}

    with database() as connection:
        cursor = connection.execute(
            """INSERT INTO jobs
               (source, chat_id, message_id, user_id, text)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (source, chat_id, message_id) DO NOTHING""",
            (
                incoming.source,
                incoming.chat_id,
                incoming.message_id,
                incoming.user_id,
                incoming.text,
            ),
        )
        job_id = cursor.lastrowid if cursor.rowcount else None

    if job_id is not None:
        background_tasks.add_task(
            process_message,
            app.state.groq,
            app.state.max_client,
            app.state.max_token,
            app.state.processing_lock,
            job_id,
        )

    return {"ok": True}
