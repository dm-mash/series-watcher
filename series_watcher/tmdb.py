"""TMDB: режиссёры и сценаристы для рекомендаций.

Связка идёт по идентификатору, а не по названию: TVmaze отдаёт
`externals.thetvdb`, по нему TMDB находит тот же сериал и отдаёт кредиты.
Без ключа источник недоступен — об этом говорится прямо, а не роняет подбор.

Форма ответов подтверждена только кодом 401 (см. PLAN.md), поэтому разбор написан
терпиво: неизвестное поле — это пустой список, а не исключение.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ENDPOINT = "https://api.themoviedb.org/3"
#: Роли приходят в разном регистре и с разными названиями, поэтому сравниваем
#: приведённые к нижнему регистру строки.
DIRECTOR_JOBS = frozenset({"director", "director of photography", "co-director"})
WRITER_JOBS = frozenset({"writer", "screenplay", "story", "author", "head writer"})
SHOW_ID_RE = re.compile(r"/tv/(\d+)")


class TmdbError(RuntimeError):
    """TMDB не ответил или ответил непонятно."""


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    role: str


@dataclass(frozen=True)
class Credits:
    directors: tuple[Person, ...] = ()
    writers: tuple[Person, ...] = ()

    def names(self) -> set[str]:
        return {person.name.casefold() for person in (*self.directors, *self.writers)}


class TmdbClient:
    name = "tmdb"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache: Any = None,
        transport: Any = None,
        timeout: float = 15.0,
        user_agent: str = "series-watcher/0.1",
    ) -> None:
        import httpx

        self._api_key = api_key or None
        self._cache = cache
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": user_agent},
        )

    def available(self) -> bool:
        return bool(self._api_key)

    def close(self) -> None:
        self._client.close()

    def find_by_tvdb(self, tvdb_id: int | None) -> int | None:
        """Найти id сериала TMDB по id TheTVDB."""
        if not tvdb_id:
            return None
        payload = self._get(f"/find/{int(tvdb_id)}", {"external_source": "tvdb_id"})
        for item in payload.get("tv_results") or ():
            value = item.get("id") if isinstance(item, Mapping) else None
            if isinstance(value, int):
                return value
        return None

    def credits(self, tmdb_id: int) -> Credits:
        payload = self._get(f"/tv/{int(tmdb_id)}/credits", {})
        crew = payload.get("crew") or ()
        directors: list[Person] = []
        writers: list[Person] = []
        for entry in crew:
            if not isinstance(entry, Mapping):
                continue
            person_id = entry.get("id")
            name = entry.get("name")
            job = entry.get("job") or ""
            if not isinstance(person_id, int) or not isinstance(name, str):
                continue
            normalized = job.casefold()
            if normalized in DIRECTOR_JOBS:
                directors.append(Person(id=person_id, name=name, role=job))
            elif normalized in WRITER_JOBS:
                writers.append(Person(id=person_id, name=name, role=job))
        return Credits(directors=tuple(directors), writers=tuple(writers))

    def _get(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        import json

        if not self.available():
            raise TmdbError("нет TMDB_API_KEY")
        query = {"api_key": self._api_key, **{k: v for k, v in params.items() if v is not None}}
        key = f"tmdb{path}?{'&'.join(f'{k}={v}' for k, v in sorted(query.items()) if k != 'api_key')}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        url = f"{ENDPOINT}{path}"
        try:
            response = self._client.get(url, params=query)
        except Exception as exc:  # noqa: BLE001 - сеть любой формы
            raise TmdbError(f"TMDB не ответил: {exc}") from exc
        if response.status_code in (401, 403):
            raise TmdbError("TMDB отказал в доступе: проверьте TMDB_API_KEY")
        if response.status_code >= 400:
            raise TmdbError(f"TMDB ответил {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise TmdbError("TMDB вернул не JSON") from exc
        if not isinstance(body, dict):
            raise TmdbError("TMDB вернул не объект")
        if self._cache is not None:
            self._cache.cache_put(key, url, json.dumps(body, ensure_ascii=False))
        return body

    def _cached(self, key: str) -> Mapping[str, Any] | None:
        import json
        import time
        from datetime import datetime

        if self._cache is None:
            return None
        entry = self._cache.cache_get(key)
        if entry is None:
            return None
        payload, fetched_at = entry
        try:
            age = time.time() - datetime.fromisoformat(fetched_at).timestamp()
        except ValueError:
            return None
        if age > 3600:
            return None
        try:
            body = json.loads(payload)
        except ValueError:
            return None
        return body if isinstance(body, dict) else None