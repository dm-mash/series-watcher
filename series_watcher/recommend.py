"""Рекомендации: авторство, жанры, объяснение, отказы.

Два сигнала, оба из TVmaze, без единого ключа:

1. авторство — кто работал над нашими сериалами и над чем ещё работал
   (`/shows/{id}/crew` и `/people/{id}/crewcredits`), с ролями: создатель весит
   больше, режиссёр и сценарист — меньше;
2. жанры — страховка, когда авторство ничего не дало.

Актёров нет: они пересекаются у всего подряд. Объяснение собирается из полей
ответа API, а не выдумывается моделью.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import Config
from .db import Database, Show
from .tvmaze import TvMazeClient, TvMazeShow

SCRIPTED = "Scripted"

CREATOR, DIRECTOR, WRITER, PRODUCER = "creator", "director", "writer", "producer"

#: Вес семейства роли в подборе.
ROLE_WEIGHTS = {CREATOR: 3.0, DIRECTOR: 2.0, WRITER: 2.0, PRODUCER: 1.0}
ROLE_LABELS = {
    CREATOR: "создатель",
    DIRECTOR: "режиссёр",
    WRITER: "сценарист",
    PRODUCER: "продюсер",
}

#: Сколько продюсеров берём с одного сериала: у сериала их десятки, а толку
#: от пятого по шестого уже нет, а запросов — на каждый отдельный.
MAX_PRODUCERS_PER_SHOW = 5


def role_kind(role: str) -> str | None:
    """Род роли из `/shows/{id}/crew`.

    В TVmaze роли гранулярные и вариантов много: у одного сериала «Executive
    Producer», «Co-Producer», «Staff Writer», «First Assistant Director». Поэтому
    сравниваем по семейству, а не по точному названию. «Director Of Photography»
    и «First Assistant Director» режиссёрами не считаются — это другие должности.
    """
    text = (role or "").casefold()
    if text == "creator":
        return CREATOR
    if text == "director":
        return DIRECTOR
    if "writer" in text or text in {"screenplay", "story", "teleplay"}:
        return WRITER
    if "producer" in text:
        return PRODUCER
    return None
DEFAULT_MIN_RATING = 6.5
#: Сколько дней расписания смотреть: три дня — это уже три запроса к TVmaze.
DEFAULT_SCHEDULE_DAYS = 3
#: Сколько сериалов разбирать подробно: каждый — это запросы к TVmaze.
DEFAULT_POOL = 8
#: Порог для холодного старта: вкуса ещё нет, поэтому берём заметно ниже.
COLD_START_MIN_RATING = 5.5


@dataclass(frozen=True)
class Recommendation:
    show: TvMazeShow
    score: float
    reasons: tuple[str, ...]
    via: str

    @property
    def summary(self) -> str | None:
        """Сводка TVmaze: из неё модель делает краткую строку о сериале."""
        return self.show.summary

    def line(self) -> str:
        meta = ", ".join(self.show.genres[:3])
        parts = [self.show.name]
        if meta:
            parts.append(f"({meta})")
        if self.show.rating is not None:
            parts.append(f"рейтинг {self.show.rating}")
        return " ".join(parts)


@dataclass(frozen=True)
class Person:
    """Человек из состава сериала: имя и семейства ролей."""

    id: int
    name: str
    roles: set[str]


@dataclass
class Recommender:
    config: Config
    db: Database
    tvmaze: TvMazeClient
    #: Даты расписания вместо ближайших дней: проверки не зависят от «сегодня».
    schedule_dates: tuple[str, ...] | None = None

    def candidates(
        self,
        *,
        limit: int = 5,
        min_rating: float = DEFAULT_MIN_RATING,
        pool_size: int = DEFAULT_POOL,
        schedule_days: int = DEFAULT_SCHEDULE_DAYS,
        focus: Show | None = None,
    ) -> list[Recommendation]:
        """Подбор по вашему списку: авторство весит больше жанров.

        `focus` — сериал, названный в запросе («похожее на Slow Horses»): тогда
        сигналы берём только по нему, а не по всей библиотеке. Исключаются всё
        равно все отслеживаемые — повторно советовать нечего.
        """
        tracked = self.db.list_shows()
        if not tracked:
            return []

        basis = [focus] if focus is not None else tracked
        excluded = {show.tvmaze_id for show in tracked} | self.db.rejected_ids()
        genres = Counter(genre for show in basis for genre in show.genres)
        authored = self._authored_shows(self._people(basis))

        scored: dict[int, Recommendation] = {}
        for show in self._pool(basis, authored, pool_size, schedule_days):
            if show.id in excluded or show.kind != SCRIPTED:
                continue
            if show.rating is not None and show.rating < min_rating:
                continue

            # Авторство — главная причина, поэтому идёт первой.
            reasons: list[str] = []
            score = 0.0
            credits = _by_weight(authored.get(show.id) or {})
            for name, roles in credits:
                for role in sorted(roles, key=lambda item: -ROLE_WEIGHTS.get(item, 0)):
                    score += ROLE_WEIGHTS.get(role, 0.0)
                    reasons.append(f"{ROLE_LABELS.get(role, role)}: {name}")

            shared = sorted(
                (genre for genre in show.genres if genres.get(genre)),
                key=lambda genre: (-genres[genre], genre),
            )
            if shared:
                score += len(shared)
                reasons.append("жанры: " + ", ".join(shared))

            if not reasons:
                continue
            scored[show.id] = Recommendation(
                show=show,
                score=score,
                reasons=tuple(dict.fromkeys(reasons)),
                via="авторство" if credits else "жанры",
            )

        ranked = sorted(scored.values(), key=lambda item: (-item.score, -(item.show.rating or 0)))
        return ranked[:limit]


    def cold_start(
        self,
        *,
        limit: int = 3,
        min_rating: float = COLD_START_MIN_RATING,
        schedule_days: int = DEFAULT_SCHEDULE_DAYS,
    ) -> list[Recommendation]:
        """Первый запуск: вкуса ещё нет, предлагаем то, что сейчас в эфире.

        Рейтинг ниже, чем при обычном подборе: иначе из расписания не проходит
        почти ничего, а показать всё равно нужно что-то.
        """
        excluded = self.db.rejected_ids() | {
            show.tvmaze_id for show in self.db.list_shows()
        }
        days = self.schedule_dates if self.schedule_dates is not None else _upcoming(schedule_days)
        items: list[Recommendation] = []
        for show in _schedule_shows(self.tvmaze, days):
            if show.id in excluded or show.kind != SCRIPTED:
                continue
            if show.rating is not None and show.rating < min_rating:
                continue
            # Рейтинг уже есть в строке кандидата, в причинах его не повторяем.
            reasons = []
            if show.genres:
                reasons.append("жанры: " + ", ".join(show.genres[:3]))
            if show.network:
                reasons.append(f"канал {show.network}")
            items.append(
                Recommendation(
                    show=show,
                    score=show.rating or 0.0,
                    reasons=tuple(reasons) or ("идёт в эфире",),
                    via="schedule",
                )
            )
        items.sort(key=lambda item: (-item.score, item.show.name))
        return items[:limit]

    # --- сигналы ---------------------------------------------------------

    def _people(self, tracked: Sequence) -> dict[int, Person]:
        """Кто работал над нашими сериалами: id человека -> имя и роли."""
        people: dict[int, Person] = {}
        for show in tracked:
            try:
                crew = self.tvmaze.get_crew(show.tvmaze_id)
            except Exception:  # noqa: BLE001 - источник не должен ронять подбор
                continue
            producers = 0
            for entry in crew:
                kind = role_kind(entry.role)
                if kind is None or not entry.person_name:
                    continue
                if kind == PRODUCER:
                    if producers >= MAX_PRODUCERS_PER_SHOW:
                        continue
                    producers += 1
                current = people.setdefault(
                    entry.person_id, Person(id=entry.person_id, name=entry.person_name, roles=set())
                )
                current.roles.add(kind)
        return people

    def _authored_shows(self, people: dict[int, Person]) -> dict[int, dict[str, set[str]]]:
        """Другие работы этих людей: id сериала -> имя -> роли."""
        result: dict[int, dict[str, set[str]]] = {}
        for person_id, person in people.items():
            try:
                credits = self.tvmaze.get_person_credits(person_id)
            except Exception:  # noqa: BLE001
                continue
            for credit in credits:
                kind = role_kind(credit.role)
                if kind is None:
                    continue
                result.setdefault(credit.show_id, {}).setdefault(person.name, set()).add(kind)
        return result

    def _pool(
        self,
        tracked: Sequence,
        authored: dict[int, dict[str, set[str]]],
        pool_size: int,
        schedule_days: int,
    ) -> list[TvMazeShow]:
        """Кандидаты: работы наших авторов плюс всё, что идёт в эфире.

        Расписание приходит уже с полными карточками сериалов, поэтому за него
        платим один запрос, а не запрос на каждый сериал.
        """
        shows: dict[int, TvMazeShow] = {}
        days = self.schedule_dates if self.schedule_dates is not None else _upcoming(schedule_days)
        for show in _schedule_shows(self.tvmaze, days):
            shows.setdefault(show.id, show)
        for tvmaze_id in list(authored) + [show.tvmaze_id for show in tracked]:
            if tvmaze_id in shows:
                continue
            try:
                shows[tvmaze_id] = self.tvmaze.get_show(tvmaze_id)
            except Exception:  # noqa: BLE001
                continue
        for show in tracked:
            shows.setdefault(show.tvmaze_id, _tracked_show(show))
        ranked = sorted(shows.values(), key=lambda item: (-(item.rating or 0.0), item.name))
        return ranked[: max(pool_size, 1)]


def _upcoming(days: int) -> list[str]:
    today = date.today()
    return [(today + timedelta(days=offset)).isoformat() for offset in range(1, days + 1)]


def _by_weight(credits: dict[str, set[str]]) -> list[tuple[str, set[str]]]:
    """Люди по убыванию своей лучшей роли: создатель выше режиссёра."""
    return sorted(
        credits.items(),
        key=lambda item: (-max((ROLE_WEIGHTS.get(role, 0.0) for role in item[1]), default=0.0), item[0]),
    )


def _schedule_shows(tvmaze: TvMazeClient, days: Iterable[str] | None = None) -> list[TvMazeShow]:
    """Сериалы из расписания. Без дней берётся всё расписание одним запросом."""
    try:
        entries = (
            tvmaze.get_schedule()
            if days is None
            else [item for day in days for item in tvmaze.get_schedule(day)]
        )
    except Exception:  # noqa: BLE001 - расписание не должно ронять подбор
        return []
    return [entry.show for entry in entries]


def _tracked_show(show) -> TvMazeShow:
    """Обратный перевод строки базы в карточку TVmaze."""
    return TvMazeShow(
        id=show.tvmaze_id,
        name=show.name,
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
        thetvdb_id=show.thetvdb_id,
        imdb_id=show.imdb_id,
        rating=show.rating,
        updated=show.source_updated,
    )