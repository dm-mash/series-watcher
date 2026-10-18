"""Источники расписания: один интерфейс, разные реализации.

Сейчас реализация одна — TVmaze. Но сервису, базе и рекомендациям всё равно:
им нужен `ScheduleSource`, который умеет найти сериал, отдать его и его эпизоды,
а по возможности — состав. Второй источник пишется рядом и подключается в
`build_source`; ломать ядро для этого не нужно.

Время эфира источник обязан привести к UTC: `airdate` и `airtime` у большинства
каталогов местные, склеивать их нельзя. Если времени нет — отдаём `None`, и
эпизод просто не попадёт в уведомления.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, Sequence

from .models import Credit, EpisodeRecord, ShowRecord, WorkCredit
from .tvmaze import TvMazeClient, TvMazeError, parse_airstamp, strip_html

#: Семейства ролей: у каталогов названия гранулярные и варьируются.
ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "creator": ("creator",),
    "director": ("director",),
    "writer": ("writer", "screenplay", "story", "teleplay"),
    "producer": ("producer",),
}

#: Похожие, но не те должности: помощник режиссёра и оператор — не режиссёр.
ROLE_EXCLUSIONS = frozenset(
    {
        "first assistant director",
        "second assistant director",
        "third assistant director",
        "second unit director",
        "director of photography",
    }
)

ROLE_LABELS = {
    "creator": "создатель",
    "director": "режиссёр",
    "writer": "сценарист",
    "producer": "продюсер",
}

ROLE_WEIGHTS = {"creator": 3.0, "director": 2.0, "writer": 2.0, "producer": 1.0}

#: Сколько продюсеров берём с одного сериала: их десятки, а толку мало.
MAX_PRODUCERS_PER_SHOW = 5


class ScheduleError(RuntimeError):
    """Источник не ответил или ответил непонятно."""


class ShowNotFound(ScheduleError):
    """Такого сериала у источника нет."""


def role_family(role: str) -> str | None:
    """«Executive Producer» → producer, «Staff Writer» → writer.

    Сравнение по семейству, а не по точному названию: у настоящего сериала
    ролей тридцать разновидностей, а «Director» в чистом виде почти не
    встречается. «First Assistant Director» и «Director Of Photography»
    режиссёрами не считаются — это другие должности.
    """
    text = (role or "").casefold().strip()
    if not text or text in ROLE_EXCLUSIONS:
        return None
    # Порядок важен: сначала самые длинные названия, иначе «Co-Producer»
    # не найдёт слова «producer» (там дефис, а не пробел).
    for family, names in ROLE_FAMILIES.items():
        for name in sorted(names, key=len, reverse=True):
            if not _mentions(text, name):
                continue
            if any(_mentions(text, other) for other in ROLE_EXCLUSIONS):
                return None
            return family
    return None


def _mentions(text: str, word: str) -> bool:
    """Есть ли слово в роли: «Co-Producer», «Staff Writer», «Director Of X»."""
    import re

    return re.search(rf"\b{re.escape(word)}\b", text) is not None


class ScheduleSource(Protocol):
    """Источник расписания. Идентификаторы — строки: не все источники числовые."""

    name: str

    def search(self, query: str) -> Sequence[ShowRecord]:
        """Найти по названию или описанию."""

    def get_show(self, external_id: str) -> ShowRecord:
        """Карточка сериала или `ShowNotFound`."""

    def get_episodes(self, external_id: str) -> Sequence[EpisodeRecord]:
        """Все известные эпизоды, с временем эфира в UTC."""

    def get_credits(self, external_id: str) -> Sequence[Credit]:
        """Состав: кто работал над сериалом. Пусто — источник такого не знает."""

    def get_works(self, person_id: str) -> Sequence[WorkCredit]:
        """Другие работы человека — основа подбора по авторству."""


class TvMazeSchedule:
    """Источник расписания на данных TVmaze.

    Разбор ответов живёт здесь: у TVmaze роли называются как попало, даты
    местные, сводки в HTML. Наружу выходит только наша модель.
    """

    name = "tvmaze"

    def __init__(self, client: TvMazeClient) -> None:
        self._client = client

    def search(self, query: str) -> Sequence[ShowRecord]:
        try:
            found = self._client.search_shows(query)
        except TvMazeError as exc:
            raise ScheduleError(str(exc)) from exc
        return [self._show(item.show) for item in found]

    def get_show(self, external_id: str) -> ShowRecord:
        try:
            return self._show(self._client.get_show(_int(external_id)))
        except TvMazeError as exc:
            raise ShowNotFound(str(exc)) from exc

    def get_episodes(self, external_id: str) -> Sequence[EpisodeRecord]:
        try:
            episodes = self._client.get_episodes(_int(external_id))
        except TvMazeError as exc:
            raise ScheduleError(str(exc)) from exc
        return [
            EpisodeRecord(
                season=item.season,
                number=item.number,
                name=item.name,
                aired_at=item.airstamp,
                runtime=item.runtime,
                summary=item.summary,
            )
            for item in episodes
        ]

    def get_credits(self, external_id: str) -> Sequence[Credit]:
        try:
            crew = self._client.get_crew(_int(external_id))
        except TvMazeError:
            return ()
        credits: list[Credit] = []
        producers = 0
        for entry in crew:
            family = role_family(entry.role)
            if family is None or not entry.person_name:
                continue
            if family == "producer":
                if producers >= MAX_PRODUCERS_PER_SHOW:
                    continue
                producers += 1
            credits.append(
                Credit(role=family, person_id=str(entry.person_id), person_name=entry.person_name)
            )
        return credits

    def get_works(self, person_id: str) -> Sequence[WorkCredit]:
        try:
            works = self._client.get_person_credits(_int(person_id))
        except TvMazeError:
            return ()
        found: list[WorkCredit] = []
        for work in works:
            family = role_family(work.role)
            if family is None:
                continue
            found.append(
                WorkCredit(
                    role=family,
                    person_id=str(work.person_id),
                    person_name=work.show_name,
                    show_external_id=str(work.show_id),
                    show_title=work.show_name,
                )
            )
        return found

    def _show(self, show) -> ShowRecord:
        identifiers = {}
        if show.thetvdb_id:
            identifiers["tvdb"] = str(show.thetvdb_id)
        if show.imdb_id:
            identifiers["imdb"] = show.imdb_id
        return ShowRecord(
            external_id=str(show.id),
            title=show.name,
            kind=show.kind,
            status=show.status,
            network=show.network,
            premiered=show.premiered,
            ended=show.ended,
            runtime=show.runtime,
            language=show.language,
            genres=tuple(show.genres),
            summary=show.summary,
            url=show.url,
            image=show.image,
            rating=show.rating,
            updated=show.updated,
            identifiers=identifiers,
        )


def build_source(config, db) -> ScheduleSource:
    """Собрать активный источник. Сегодня он один, но выбор уже тут."""
    from .tvmaze import TvMazeClient

    client = TvMazeClient(
        base_url=config.tvmaze_base_url,
        cache=db,
        timeout=config.http_timeout,
        retries=config.http_retries,
        ttl=config.tvmaze_cache_ttl,
        user_agent=config.user_agent,
        rate_limit=config.tvmaze_rate_limit,
    )
    return TvMazeSchedule(client)


def clean_html(value: str | None) -> str | None:
    """Сводки приходят в HTML; в модель уходит уже текст."""
    return strip_html(value)


def utc(value: datetime | None) -> datetime | None:
    return value.astimezone(timezone.utc) if value else None


def _int(value: str | int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScheduleError(f"идентификатор должен быть числом, получено {value!r}") from exc