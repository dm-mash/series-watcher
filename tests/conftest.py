"""Общие приспособления для тестов.

Тесты не ходят в сеть: api.tvmaze.com закрыт `httpx.MockTransport`, ответы
берутся из сохранённых реальных фикстур в `tests/fixtures/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"

APP_ENV_VARS = (
    "TZ",
    "DB_PATH",
    "HTTP_TIMEOUT",
    "HTTP_RETRIES",
    "USER_AGENT",
    "TVMAZE_BASE_URL",
    "TVMAZE_CACHE_TTL",
    "TVMAZE_RATE_LIMIT",
    "NOTIFY_LOOKBACK_HOURS",
    "QUIET_HOURS",
    "JUSTWATCH_COUNTRY",
    "CUSTOM_SOURCES_PATH",
    "TMDB_API_KEY",
    "JUSTWATCH_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ни один настоящий ключ из `.env` не должен попасть в тест."""
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TZ", "UTC")


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def real_routes() -> dict[str, Any]:
    """Маршруты из сохранённых ответов TVmaze."""
    return {
        "/search/shows?q=Severance": load_fixture("search_shows"),
        "/shows/44933": load_fixture("show"),
        "/shows/44933/episodes": load_fixture("episodes"),
        "/shows/75194": load_fixture("show_running"),
        "/shows/75194/episodes": load_fixture("episodes_running"),
        "/schedule?date=2026-10-19": load_fixture("schedule"),
    }


class FakeTVmaze:
    """Подмена api.tvmaze.com по маршрутам и счётчик запросов."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes: dict[str, Any] = dict(routes or real_routes())
        self.calls: list[str] = []

    def add(self, key: str, payload: Any, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.routes[key] = _Route(payload, status, headers or {})

    def calls_to(self, key: str) -> int:
        return sum(1 for call in self.calls if call == key or call.startswith(f"{key}&"))

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            query = request.url.query.decode()
            key = f"{path}?{query}" if query else path
            self.calls.append(key)
            plain = f"{path}?{unquote_plus(query)}" if query else path
            route = self.routes.get(key, self.routes.get(plain, self.routes.get(path)))
            if route is None:
                return httpx.Response(404, json={"name": "Not found"})
            route = route if isinstance(route, _Route) else _Route(route, 200, {})
            return httpx.Response(route.status, json=route.payload, headers=route.headers)

        return httpx.MockTransport(handler)


class _Route:
    def __init__(self, payload: Any, status: int, headers: dict[str, str]) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers


class FakeTelegram:
    """Подмена api.telegram.org: собирает отправленные сообщения."""

    def __init__(self, *, status: int = 200, ok: bool = True, description: str = "") -> None:
        self.status = status
        self.ok = ok
        self.description = description
        self.sent: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.sent.append(json.loads(request.content.decode("utf-8")))
            if self.status >= 400:
                return httpx.Response(self.status, text="telegram is down")
            body: dict[str, Any] = {"ok": self.ok, "result": {"message_id": len(self.sent)}}
            if not self.ok:
                body["description"] = self.description or "Bad Request"
            return httpx.Response(self.status, json=body)

        return httpx.MockTransport(handler)


class FakeGraphQL:
    """Подмена GraphQL-эндпоинта источника: один ответ на все запросы."""

    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.queries: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            self.queries.append(body)
            return httpx.Response(self.status, json=self.payload)

        return httpx.MockTransport(handler)


@pytest.fixture
def fake_tvmaze() -> FakeTVmaze:
    return FakeTVmaze()


@pytest.fixture
def sleeps() -> list[float]:
    return []


def pin_schedule(fake, payload=None, *, days: int = 3) -> None:
    """Расписание живёт в кэше по ключу «дата», поэтому вешаем и его, и даты.

    Так тесты не зависят от того, какой сегодня день, и от того, спрашивает ли
    стратегия расписание целиком или по дням.
    """
    import datetime

    if payload is None:
        payload = load_fixture("schedule")
    fake.add("/schedule", payload)
    today = datetime.date.today()
    for offset in range(1, days + 1):
        day = (today + datetime.timedelta(days=offset)).isoformat()
        fake.add(f"/schedule?date={day}", payload)


def store_show(db, payload: dict, source: str = "tvmaze"):
    """Положить карточку источника в базу так же, как это делает сервис."""
    from series_watcher.schedule import TvMazeSchedule
    from series_watcher.tvmaze import parse_show

    record = TvMazeSchedule(None)._show(parse_show(payload))
    return db.upsert_show(source, record)