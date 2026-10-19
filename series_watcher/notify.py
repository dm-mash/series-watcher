"""Telegram: отправка, дедупликация, тихий режим.

Отметка об отправке ставится только после успешного ответа Telegram: иначе
эпизод потерялся бы навсегда. Без токена отправка не происходит, и об этом
говорится прямо, а не падает.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from .config import Config, in_quiet_hours
from .db import TELEGRAM_CHANNEL, CHAT_ID_SETTING, Database, utcnow
from .tvmaze import TvMazeError

TVMAZE_CREDIT = "Данные: TVmaze (CC BY-SA 4.0), https://www.tvmaze.com"

#: Запас поверх long polling: сетевые задержки не должны рвать соединение.
LONGPOLL_MARGIN = 15.0


class NotifyError(RuntimeError):
    """Telegram не принял сообщение."""


@dataclass(frozen=True)
class SendResult:
    sent: bool
    detail: str = ""


def esc(text: object) -> str:
    """Экранирование для HTML-разметки Telegram.

    Всё, что пришло извне — имена сериалов, названия раздач, сводки — обязано
    пройти через это: иначе один `&` в названии ломает всё сообщение.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_TAG = re.compile(r"</?(?:b|i|code|s|u)>")


def as_text(html: str) -> str:
    """Разметка → plain text: так тот же ответ читается в терминале.

    Теги снимаются, ссылки остаются: в терминале `magnet:?xt=…` полезнее слова
    «magnet», а нажимать по ней всё равно нельзя.
    """
    return _TAG.sub("", html)


def footer(extra: Sequence[str] = ()) -> str:
    """Подпись с атрибуцией: без неё ответ бота нарушает условия TVmaze."""
    return "\n".join([*extra, TVMAZE_CREDIT])


def credit() -> str:
    """Атрибуция помельче: курсивом и вплотную к последней строке.

    Подпись обязательна, но она не должна спорить с ответом: раньше она стояла
    отдельным абзацем и читалась как ещё один пункт списка.
    """
    return f"\n<i>{TVMAZE_CREDIT}</i>"


def footer_credit(extra: Sequence[str] = ()) -> str:
    """Ответ с атрибуцией — строкой помельче, вплотную к последней строке."""
    body = [line for line in extra if line]
    if not body:
        return TVMAZE_CREDIT
    return "\n".join([*body, credit()])


class Notifier:
    """Отправка в Telegram с отметками о доставке."""

    def __init__(
        self,
        config: Config,
        db: Database,
        *,
        transport: Any = None,
        client: Any = None,
        token: str | None = None,
        chat_id: int | str | None = None,
    ) -> None:
        import httpx

        self._config = config
        self._db = db
        self._token = token if token is not None else config.telegram_bot_token
        self._chat = self._resolve_chat(config, chat_id, db)
        self._client = client or httpx.Client(
            timeout=config.http_timeout,
            transport=transport,
            headers={"User-Agent": config.user_agent},
        )
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat)

    @staticmethod
    def _resolve_chat(config: Config, chat_id: int | str | None, db: Database) -> str | None:
        """Явный `TELEGRAM_CHAT_ID` важнее; иначе — откуда бот последний раз писал."""
        if chat_id is not None:
            return str(chat_id)
        if config.telegram_chat_id:
            return str(config.telegram_chat_id)
        return db.get_setting(CHAT_ID_SETTING)

    @property
    def client(self) -> Any:
        return self._client

    @property
    def channel(self) -> str:
        return TELEGRAM_CHANNEL

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def is_quiet(self, moment: datetime | None = None) -> bool:
        now = moment or utcnow()
        return in_quiet_hours(now.astimezone(self._config.tz).hour, self._config.notify_quiet_hours)

    def send(self, text: str, *, html: bool = False) -> SendResult:
        """Отправить текст. Без токена или чата — отказ, а не исключение."""
        if not self.configured:
            return SendResult(sent=False, detail="нет TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
        try:
            TelegramApi(self._config, self._client, token=self._token).send_message(
                self._chat, text, html=html
            )
        except (NotifyError, TvMazeError) as exc:
            return SendResult(sent=False, detail=str(exc))
        return SendResult(sent=True)

    def send_episode(
        self, episode_id: int, text: str, *, kind: str = "new_episode", html: bool = False
    ) -> SendResult:
        """Отправить и только потом отметить: иначе эпизод потерялся бы."""
        if self._db.is_sent(episode_id, self.channel, kind):
            return SendResult(sent=False, detail="уже отправлено")
        result = self.send(text, html=html)
        if result.sent:
            self._db.mark_sent(episode_id, self.channel, kind)
        return result


class TelegramApi:
    """Тонкая обвязка Bot API: отправка и разбор входящих."""

    def __init__(self, config: Config, client: Any, *, token: str | None = None) -> None:
        self._config = config
        self._client = client
        self._token = token or config.telegram_bot_token

    @property
    def token(self) -> str | None:
        return self._token

    def send_message(
        self,
        chat_id: Any,
        text: str,
        *,
        buttons: Sequence[Sequence[str]] = (),
        html: bool = False,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if html:
            # Разметка приходит из наших же строк, всё, что от API, экранировано
            # заранее. HTML, а не Markdown: там нужно бегать по `_` и `*` в именах
            # файлов раздач, а здесь — только по трём символам.
            payload["parse_mode"] = "HTML"
        keyboard = _keyboard(buttons)
        if keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        return self._call("sendMessage", payload)

    def get_updates(self, *, offset: int | None = None, timeout: int = 25) -> list[Mapping[str, Any]]:
        """Долгий опрос: соединение держится `timeout` секунд, поэтому и HTTP-таймаут
        должен быть длиннее, иначе клиент отвалится раньше ответа Telegram."""
        payload: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload, http_timeout=timeout + LONGPOLL_MARGIN)
        return [item for item in (result or []) if isinstance(item, Mapping)]

    def close(self) -> None:
        self._client.close()

    def _call(
        self, method: str, payload: Mapping[str, Any], *, http_timeout: float | None = None
    ) -> Any:
        import httpx

        if not self._token:
            raise NotifyError("нет TELEGRAM_BOT_TOKEN")
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            response = self._client.post(
                url, json=dict(payload), **({"timeout": http_timeout} if http_timeout else {})
            )
        except httpx.HTTPError as exc:
            raise NotifyError(f"Telegram не ответил: {exc}") from exc
        if response.status_code >= 400:
            raise NotifyError(f"Telegram ответил HTTP {response.status_code}: {response.text[:200]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise NotifyError("Telegram вернул не JSON") from exc
        if isinstance(body, Mapping) and body.get("ok") is False:
            description = body.get("description") or "без описания"
            raise NotifyError(f"Telegram отказал: {description}")
        return body.get("result") if isinstance(body, Mapping) else None


def _keyboard(buttons: Sequence[Sequence[Any]]) -> list[list[dict[str, str]]]:
    """Ряды кнопок Telegram.

    Ряд — это кортеж кнопок, кнопка — пара `(подпись, callback)`. Ряд из одной
    кнопки пишется как `((подпись, callback),)`.

    Подпись вместо пары тоже можно: нажатие отправляет её как реплику. Но ряд
    ровно из двух подписей неотличим от одной пары, поэтому такую ошибку лучше
    поймать здесь, чем отправить в Telegram две кнопки: у первой callback будет
    равен подписи, и нажатие ничего не сделает.
    """
    rows: list[list[dict[str, str]]] = []
    for row in buttons:
        if len(row) == 2 and all(isinstance(button, str) for button in row):
            raise TypeError(
                f"ряд кнопок {row!r}: это одна кнопка или две подписки? "
                "кнопка пишется как ((подпись, callback),)"
            )
        keyboard: list[dict[str, str]] = []
        for button in row:
            if isinstance(button, (tuple, list)):
                label, callback = button
                keyboard.append({"text": str(label), "callback_data": str(callback)})
            else:
                keyboard.append({"text": str(button), "callback_data": str(button)})
        if keyboard:
            rows.append(keyboard)
    return rows