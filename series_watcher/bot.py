"""Разговорный слой: сообщения Telegram -> намерение -> ответ."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .config import Config
from .llm import LLMClient
from .notify import TelegramApi
from .nlu import Intent, parse
from .service import Reply, SeriesWatcher


@dataclass(frozen=True)
class HandledMessage:
    update_id: int
    chat_id: Any
    text: str
    reply: str
    buttons: tuple[tuple[str, ...], ...] = ()


class Bot:
    """Разговорный слой поверх сервиса.

    Каждая реплика самодостаточна: история диалога не хранится, ответ строится
    запросом к базе. Кнопки — запасной путь, когда разбор неуверенный.
    """

    def __init__(
        self,
        config: Config,
        watcher: SeriesWatcher,
        api: TelegramApi,
        *,
        llm: LLMClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._watcher = watcher
        self._api = api
        self._llm = llm
        self._sleep = sleep
        self._offset: int | None = None

    @property
    def usable(self) -> bool:
        """Есть ли вообще куда спросить: без токена бот — пустой цикл."""
        return bool(self._api.token)

    def understand(self, text: str) -> Intent:
        return parse(text, client=self._llm)

    def answer(self, text: str) -> Reply:
        intent = self.understand(text)
        return self._watcher.handle(intent)

    def handle_update(self, update: Mapping[str, Any]) -> HandledMessage | None:
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, Mapping):
            return None
        text = message.get("text")
        chat = message.get("chat") or {}
        if not isinstance(text, str) or not text.strip() or not isinstance(chat, Mapping):
            return None
        reply = self.answer(text)
        self._api.send_message(chat.get("id"), reply.text, buttons=reply.buttons)
        return HandledMessage(
            update_id=int(update.get("update_id") or 0),
            chat_id=chat.get("id"),
            text=text,
            reply=reply.text,
            buttons=reply.buttons,
        )

    def poll_once(self, *, timeout: int = 25) -> list[HandledMessage]:
        updates = self._api.get_updates(offset=self._offset, timeout=timeout)
        handled: list[HandledMessage] = []
        for update in updates:
            self._offset = int(update.get("update_id") or 0) + 1
            message = self.handle_update(update)
            if message is not None:
                handled.append(message)
        return handled

    def run(self, *, once: bool = False, poll_timeout: int = 25) -> None:
        """Цикл опроса. `once` — один заход, удобно для cron и проверок."""
        while True:
            handled = self.poll_once(timeout=poll_timeout)
            if once:
                return
            if not handled:
                continue


def text_update(text: str, *, chat_id: Any = 1, update_id: int = 1) -> dict[str, Any]:
    """Одно обновление с текстовым сообщением — для проверок."""
    return {
        "update_id": update_id,
        "message": {"message_id": update_id, "chat": {"id": chat_id}, "text": text},
    }