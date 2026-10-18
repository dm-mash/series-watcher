"""Рекомендации: авторство, жанры, объяснение, отказы.

Три уровня сигналов, и подбор работает при любом наборе ключей:

1. `Creator` из TVmaze — есть почти всегда, ключ не нужен;
2. режиссёры и сценаристы из TMDB — точная находка, нужен ключ;
3. жанры — страховка, когда ключа нет или сериал в TMDB не нашёлся.

Актёров нет: они пересекаются у всего подряд. Объяснение собирается из полей
ответа API, а не выдумывается моделью.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import Config
from .db import Database
from .tmdb import TmdbClient, TmdbError
from .tvmaze import CREATOR_ROLE, TvMazeClient, TvMazeShow

SCRIPTED = "Scripted"
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

    def line(self) -> str:
        meta = ", ".join(self.show.genres[:3])
        parts = [self.show.name]
        if meta:
            parts.append(f"({meta})")
        if self.show.rating is not None:
            parts.append(f"рейтинг {self.show.rating}")
        return " ".join(parts)


@dataclass
class Recommender:
    config: Config
    db: Database
    tvmaze: TvMazeClient
    tmdb: TmdbClient | None = None
    #: Даты расписания вместо ближайших дней: проверки не зависят от «сегодня».
    schedule_dates: tuple[str, ...] | None = None

    def candidates(
        self,
        *,
        limit: int = 5,
        min_rating: float = DEFAULT_MIN_RATING,
        pool_size: int = DEFAULT_POOL,
        use_tmdb: bool = True,
        schedule_days: int = DEFAULT_SCHEDULE_DAYS,
    ) -> list[Recommendation]:
        tracked = self.db.list_shows()
        if not tracked:
            return []

        excluded = {show.tvmaze_id for show in tracked} | self.db.rejected_ids()
        genres = Counter(genre for show in tracked for genre in show.genres)
        creators = self._creators(tracked)
        creator_shows = self._creator_shows(creators)
        credits = self._tmdb_credits(tracked) if use_tmdb else {}

        scored: dict[int, Recommendation] = {}
        for show in self._pool(tracked, creator_shows, pool_size, schedule_days):
            if show.id in excluded or show.kind != SCRIPTED:
                continue
            if show.rating is not None and show.rating < min_rating:
                continue

            reasons: list[str] = []
            score = 0.0
            shared = sorted(
                (genre for genre in show.genres if genres.get(genre)),
                key=lambda genre: (-genres[genre], genre),
            )
            if shared:
                score += len(shared)
                reasons.append("жанры: " + ", ".join(shared))

            author = creator_shows.get(show.id)
            if author:
                score += 3.0
                reasons.append("создатель: " + ", ".join(sorted(author)))

            matched = self._credits_match(show, credits) if use_tmdb else ()
            if matched:
                score += 2.0
                reasons.append("режиссёры и сценаристы: " + ", ".join(matched))

            if not reasons:
                continue
            scored[show.id] = Recommendation(
                show=show,
                score=score,
                reasons=tuple(reasons),
                via="creator" if author else ("tmdb" if matched else "genre"),
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

    def _creators(self, tracked: Sequence) -> dict[int, str]:
        """Кто создаёт наши сериалы: id человека -> имя."""
        creators: dict[int, str] = {}
        for show in tracked:
            try:
                crew = self.tvmaze.get_crew(show.tvmaze_id)
            except Exception:  # noqa: BLE001 - источник не должен ронять подбор
                continue
            for entry in crew:
                if entry.role == CREATOR_ROLE:
                    creators[entry.person_id] = entry.person_name
        return creators

    def _creator_shows(self, creators: dict[int, str]) -> dict[int, set[str]]:
        """Другие сериалы тех же создателей: id сериала -> имена авторов."""
        result: dict[int, set[str]] = {}
        for person_id, person_name in creators.items():
            try:
                credits = self.tvmaze.get_person_credits(person_id)
            except Exception:  # noqa: BLE001
                continue
            for credit in credits:
                if credit.role == CREATOR_ROLE:
                    result.setdefault(credit.show_id, set()).add(person_name)
        return result

    def _tmdb_credits(self, tracked: Sequence) -> dict[int, tuple[str, ...]]:
        """Имена режиссёров и сценаристов по сериалам: id TVmaze -> имена."""
        if self.tmdb is None or not self.tmdb.available():
            return {}
        credits: dict[int, tuple[str, ...]] = {}
        for show in tracked:
            try:
                tmdb_id = self.tmdb.find_by_tvdb(show.thetvdb_id)
                if tmdb_id is None:
                    continue
                found = self.tmdb.credits(tmdb_id)
            except TmdbError:
                continue
            names = tuple(sorted(found.names()))
            if names:
                credits[show.tvmaze_id] = names
        return credits

    def _pool(
        self,
        tracked: Sequence,
        creator_shows: dict[int, set[str]],
        pool_size: int,
        schedule_days: int,
    ) -> list[TvMazeShow]:
        """Кандидаты: работы создателей плюс всё, что идёт в эфире.

        Расписание приходит уже с полными карточками сериалов, поэтому за него
        платим один запрос, а не запрос на каждый сериал.
        """
        shows: dict[int, TvMazeShow] = {}
        days = self.schedule_dates if self.schedule_dates is not None else _upcoming(schedule_days)
        for show in _schedule_shows(self.tvmaze, days):
            shows.setdefault(show.id, show)
        for tvmaze_id in creator_shows:
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

    def _credits_match(self, show: TvMazeShow, credits: dict[int, tuple[str, ...]]) -> tuple[str, ...]:
        wanted = {name for names in credits.values() for name in names}
        if not wanted:
            return ()
        try:
            crew = self.tvmaze.get_crew(show.id)
        except Exception:  # noqa: BLE001
            return ()
        return tuple(
            sorted({entry.person_name.casefold() for entry in crew if entry.person_name.casefold() in wanted})
        )


def _upcoming(days: int) -> list[str]:
    today = date.today()
    return [(today + timedelta(days=offset)).isoformat() for offset in range(1, days + 1)]


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