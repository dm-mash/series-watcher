"""Клиент TVmaze: расписание, поиск, каталог.

Лимит TVmaze — 20 запросов на 10 секунд на IP, поэтому запросы проходят через
окно RateLimiter, а ответы кладутся в локальный кэш: по полю `updated` у
сериала и по TTL у списка эпизодов (в `/shows/{id}/episodes` поля `updated`
нет — список отдаётся целиком и без метки актуальности).
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Mapping, Protocol

import httpx

DEFAULT_BASE_URL = "https://api.tvmaze.com"
DEFAULT_TTL = 21600.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_SLEEP = 10.0
_SHOW_HREF_RE = re.compile(r"/shows/(\d+)")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ARTICLE_PREFIXES = ("the ", "a ", "an ")


class TvMazeError(RuntimeError):
    """Ответ TVmaze не годится."""


class TvMazeNotFound(TvMazeError):
    """Сериала или эпизода нет."""


class TvMazeRateLimited(TvMazeError):
    """Лимит запросов исчерпан и повторы не помогли."""


class HttpCache(Protocol):
    """То, что `db.Database` умеет отдавать клиенту."""

    def cache_get(self, key: str) -> tuple[str, str] | None: ...

    def cache_put(self, key: str, url: str, payload: str) -> None: ...


def strip_html(value: str | None) -> str | None:
    """Сводки TVmaze приходят в HTML, а хранить и показывать их нужно текстом."""
    if not value:
        return None
    text = _TAG_RE.sub(" ", value)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = _WS_RE.sub(" ", text)
    text = text.strip()
    return text or None


def normalize_name(value: str) -> str:
    """Ключ для сравнения названий: без регистра, артиклей и лишних пробелов."""
    text = _WS_RE.sub(" ", value or "").strip().casefold()
    for prefix in _ARTICLE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip(" -:,!?")


def parse_airstamp(value: Any) -> datetime | None:
    """`airstamp` — единственный источник времени эфира.

    `airdate` и `airtime` у TVmaze местные для страны показа: эпизод с
    `airdate: 2026-10-19, airtime: 00:00` выходит в `airstamp` как
    `2026-10-20T04:00:00+00:00`. Склеивать дату со временем нельзя.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class TvMazeShow:
    id: int
    name: str
    kind: str | None = None
    status: str | None = None
    network: str | None = None
    premiered: str | None = None
    ended: str | None = None
    runtime: int | None = None
    language: str | None = None
    genres: tuple[str, ...] = ()
    summary: str | None = None
    url: str | None = None
    image: str | None = None
    thetvdb_id: int | None = None
    imdb_id: str | None = None
    rating: float | None = None
    updated: int | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


@dataclass(frozen=True)
class TvMazeEpisode:
    season: int
    number: int
    show_id: int
    id: int | None = None
    name: str | None = None
    airstamp: datetime | None = None
    airdate: str | None = None
    airtime: str | None = None
    runtime: int | None = None
    summary: str | None = None


@dataclass(frozen=True)
class TvMazeCrew:
    """Запись из `/shows/{id}/crew`: роль и человек."""

    show_id: int
    role: str
    person_id: int
    person_name: str


@dataclass(frozen=True)
class TvMazeCredit:
    """Работа человека: из `/people/{id}/crewcredits`."""

    person_id: int
    role: str
    show_id: int
    show_name: str


@dataclass(frozen=True)
class SearchResult:
    score: float
    show: TvMazeShow


@dataclass(frozen=True)
class ScheduleEntry:
    show: TvMazeShow
    episode: TvMazeEpisode


def parse_show(payload: Mapping[str, Any]) -> TvMazeShow:
    network = payload.get("network") or payload.get("webChannel") or {}
    externals = payload.get("externals") or {}
    image = payload.get("image") or {}
    rating = payload.get("rating") or {}
    updated = payload.get("updated")
    return TvMazeShow(
        id=int(payload["id"]),
        name=str(payload.get("name") or ""),
        kind=payload.get("type"),
        status=payload.get("status"),
        network=network.get("name"),
        premiered=payload.get("premiered"),
        ended=payload.get("ended"),
        runtime=payload.get("averageRuntime") or payload.get("runtime"),
        language=payload.get("language"),
        genres=tuple(payload.get("genres") or ()),
        summary=strip_html(payload.get("summary")),
        url=payload.get("url"),
        image=image.get("medium") or image.get("original"),
        thetvdb_id=externals.get("thetvdb"),
        imdb_id=externals.get("imdb"),
        rating=rating.get("average"),
        updated=int(updated) if isinstance(updated, (int, float)) else None,
    )


def parse_episode(payload: Mapping[str, Any], show_id: int) -> TvMazeEpisode:
    return TvMazeEpisode(
        season=int(payload.get("season") or 0),
        number=int(payload.get("number") or 0),
        show_id=show_id,
        id=payload.get("id"),
        name=payload.get("name"),
        airstamp=parse_airstamp(payload.get("airstamp")),
        airdate=payload.get("airdate"),
        airtime=payload.get("airtime") or None,
        runtime=payload.get("runtime"),
        summary=strip_html(payload.get("summary")),
    )


def dedupe_episodes(episodes: Iterable[TvMazeEpisode]) -> list[TvMazeEpisode]:
    """Оставить по одному эпизоду на `(season, number)` — это ключ дедупликации."""
    seen: dict[tuple[int, int], TvMazeEpisode] = {}
    for episode in episodes:
        seen[(episode.season, episode.number)] = episode
    return [seen[key] for key in sorted(seen)]


class RateLimiter:
    """Скользящее окно: не больше `max_requests` запросов за `window` секунд."""

    def __init__(
        self,
        max_requests: int = 18,
        window: float = 10.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_requests = max(0, int(max_requests))
        self.window = float(window)
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        if self.max_requests == 0:
            return
        now = self._clock()
        self._forget(now)
        if len(self._stamps) >= self.max_requests:
            wait = self.window - (now - self._stamps[0]) + 0.01
            if wait > 0:
                self._sleep(min(wait, MAX_SLEEP))
                now = self._clock()
                self._forget(now)
        self._stamps.append(now)

    def _forget(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] >= self.window:
            self._stamps.popleft()


class TvMazeClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache: HttpCache | None = None,
        transport: Any = None,
        timeout: float = 15.0,
        retries: int = 3,
        ttl: float = DEFAULT_TTL,
        user_agent: str = "series-watcher/0.1",
        rate_limit: tuple[int, float] | None = (18, 10.0),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep = sleep
        self._cache = cache
        self._ttl = ttl
        self._retries = max(0, int(retries))
        self._timeout = timeout
        self._user_agent = user_agent
        if rate_limit is None:
            self._limiter = RateLimiter(0, 10.0, clock=clock, sleep=sleep)
        else:
            self._limiter = RateLimiter(*rate_limit, clock=clock, sleep=sleep)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )
        self.request_count = 0

    # --- публичный интерфейс -------------------------------------------

    def search_shows(self, query: str, *, refresh: bool = False) -> list[SearchResult]:
        payload = self._get("/search/shows", {"q": query}, refresh=refresh)
        results: list[SearchResult] = []
        for item in payload or ():
            show_payload = item.get("show")
            if not show_payload:
                continue
            results.append(
                SearchResult(score=float(item.get("score") or 0.0), show=parse_show(show_payload))
            )
        results.sort(key=lambda item: (-item.score, item.show.id))
        return results

    def get_show(self, show_id: int, *, refresh: bool = False) -> TvMazeShow:
        payload = self._get(f"/shows/{show_id}", refresh=refresh)
        return parse_show(payload)

    def get_episodes(self, show_id: int, *, refresh: bool = False) -> list[TvMazeEpisode]:
        payload = self._get(f"/shows/{show_id}/episodes", refresh=refresh)
        return dedupe_episodes(parse_episode(item, show_id) for item in payload or ())

    def get_crew(self, show_id: int, *, refresh: bool = False) -> list[TvMazeCrew]:
        """Группа сериала: роли вроде `Creator` живут именно здесь."""
        payload = self._get(f"/shows/{show_id}/crew", refresh=refresh)
        crew: list[TvMazeCrew] = []
        for item in payload or ():
            person = (item or {}).get("person") or {}
            role = (item or {}).get("type")
            person_id = person.get("id")
            if not role or not isinstance(person_id, int):
                continue
            crew.append(
                TvMazeCrew(
                    show_id=show_id,
                    role=str(role),
                    person_id=person_id,
                    person_name=str(person.get("name") or ""),
                )
            )
        return crew

    def get_person_credits(self, person_id: int, *, refresh: bool = False) -> list[TvMazeCredit]:
        """Все работы человека: другие сериалы приходят отсюда."""
        payload = self._get(f"/people/{int(person_id)}/crewcredits", refresh=refresh)
        credits: list[TvMazeCredit] = []
        for item in payload or ():
            role = (item or {}).get("type")
            link = ((item or {}).get("_links") or {}).get("show") or {}
            href = str(link.get("href") or "")
            match = _SHOW_HREF_RE.search(href)
            if not role or match is None:
                continue
            credits.append(
                TvMazeCredit(
                    person_id=person_id,
                    role=str(role),
                    show_id=int(match.group(1)),
                    show_name=str(link.get("name") or ""),
                )
            )
        return credits

    def get_schedule(self, date: str | None = None, *, refresh: bool = False) -> list[ScheduleEntry]:
        """Расписание на день. Без даты TVmaze отдаёт всё, что идёт сейчас."""
        params = {"date": date} if date else None
        payload = self._get("/schedule", params, refresh=refresh)
        entries: list[ScheduleEntry] = []
        for item in payload or ():
            show_payload = item.get("show")
            if not show_payload:
                continue
            show = parse_show(show_payload)
            entries.append(ScheduleEntry(show=show, episode=parse_episode(item, show.id)))
        return entries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TvMazeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- транспорт ------------------------------------------------------

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        refresh: bool = False,
        ttl: float | None = None,
    ) -> Any:
        key = cache_key(path, params)
        if not refresh:
            cached = self._cached(key, ttl if ttl is not None else self._ttl)
            if cached is not None:
                return cached
        url = self._client.base_url.join(path)
        data = self._request(str(url), params)
        if self._cache is not None:
            self._cache.cache_put(key, str(url), json.dumps(data, ensure_ascii=False))
        return data

    def _cached(self, key: str, ttl: float) -> Any | None:
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
        if ttl <= 0 or age > ttl:
            return None
        return json.loads(payload)

    def _request(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            self._limiter.acquire()
            try:
                self.request_count += 1
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == self._retries:
                    break
                self._sleep(_backoff(attempt))
                continue

            if response.status_code == 404:
                raise TvMazeNotFound(f"TVmaze не знает {url}")
            if response.status_code in RETRY_STATUSES:
                last_error = TvMazeError(f"TVmaze ответил {response.status_code} на {url}")
                if attempt == self._retries:
                    break
                self._sleep(_retry_after(response, attempt))
                continue
            if response.status_code >= 400:
                raise TvMazeError(f"TVmaze ответил {response.status_code} на {url}")
            try:
                return response.json()
            except ValueError as exc:
                raise TvMazeError(f"TVmaze вернул не JSON на {url}: {exc}") from exc

        if isinstance(last_error, TvMazeError) and "429" in str(last_error):
            raise TvMazeRateLimited(str(last_error)) from last_error
        raise TvMazeError(f"TVmaze не ответил на {url}: {last_error}") from last_error


def cache_key(path: str, params: Mapping[str, Any] | None) -> str:
    if not params:
        return path
    query = "&".join(f"{name}={value}" for name, value in sorted(params.items()) if value)
    return f"{path}?{query}" if query else path


def _backoff(attempt: int) -> float:
    return min(MAX_SLEEP, 0.5 * (2**attempt))


def _retry_after(response: Any, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return min(MAX_SLEEP, float(raw))
        except ValueError:
            try:
                when = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                return _backoff(attempt)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return min(MAX_SLEEP, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
    return _backoff(attempt)

#: Роль, по которой сериал считают «тем же авторством».
CREATOR_ROLE = "Creator"
