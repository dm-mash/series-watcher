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
    "TMDB_API_KEY",
    "JUSTWATCH_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
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


@pytest.fixture
def fake_tvmaze() -> FakeTVmaze:
    return FakeTVmaze()


@pytest.fixture
def sleeps() -> list[float]:
    return []