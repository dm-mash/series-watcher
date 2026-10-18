"""Сервис: что делает бот и CLI поверх базы и источников.

Пока есть добавление сериала и список отслеживаемого. Опрос расписания и
уведомления добавляются следующим этапом, дедупликация там опирается на
`(show_id, season, number)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import Config
from .db import Database, Episode, Show, TrackedShow, tracked_shows, utcnow
from .tvmaze import TvMazeClient, TvMazeError, TvMazeShow, normalize_name

#: Канал один: сервис рассчитан на одного пользователя в Telegram.
NOTIFY_CHANNEL = "telegram"


@dataclass(frozen=True)
class Candidate:
    show: TvMazeShow
    score: float

    @property
    def label(self) -> str:
        meta = ", ".join(part for part in (self.show.kind, self.show.status, self.show.network) if part)
        return f"{self.show.name} ({meta})" if meta else self.show.name


@dataclass(frozen=True)
class AddResult:
    status: str  # added | exists | needs_choice | not_found
    show: Show | None = None
    candidates: tuple[Candidate, ...] = ()
    episodes: int = 0
    message: str = ""


@dataclass(frozen=True)
class DueEpisode:
    """Эпизод, который уже вышел и о котором ещё не сообщали."""

    show: Show
    episode: Episode


@dataclass(frozen=True)
class ShowPoll:
    show: Show
    episodes_total: int
    new_episodes: int
    updated_episodes: int
    without_time: int
    error: str | None = None


@dataclass(frozen=True)
class PollReport:
    checked_at: datetime
    shows: tuple[ShowPoll, ...]
    due: tuple[DueEpisode, ...]

    @property
    def failed(self) -> tuple[ShowPoll, ...]:
        return tuple(item for item in self.shows if item.error)

    @property
    def new_episodes(self) -> int:
        return sum(item.new_episodes for item in self.shows)


@dataclass
class SeriesWatcher:
    config: Config
    db: Database
    tvmaze: TvMazeClient

    def add_show(self, query: str, *, pick: int | None = None, limit: int = 7) -> AddResult:
        """Найти сериал в TVmaze и завести его в списке.

        Название совпало — добавляем сразу. Иначе возвращаем кандидатов, чтобы
        выбор сделал человек: поиск TVmaze идёт по подстроке, и «office» находит
        и «The Office», и «Offices».
        """
        needle = normalize_name(query)
        if not needle:
            return AddResult(status="not_found", message="Пустой запрос")

        found = self.tvmaze.search_shows(query)
        candidates = tuple(Candidate(show=item.show, score=item.score) for item in found[:limit])
        if not candidates:
            return AddResult(status="not_found", message=f"По запросу «{query}» ничего нет")

        exact = next((item for item in candidates if item.show.normalized_name == needle), None)
        chosen: TvMazeShow | None
        if exact is not None:
            chosen = exact.show
        elif pick is not None:
            if pick < 1 or pick > len(candidates):
                return AddResult(
                    status="needs_choice",
                    candidates=candidates,
                    message=f"Нет варианта №{pick}, их {len(candidates)}",
                )
            chosen = candidates[pick - 1].show
        elif len(candidates) == 1:
            chosen = candidates[0].show
        else:
            return AddResult(
                status="needs_choice",
                candidates=candidates,
                message=f"Под «{query}» нашлось несколько сериалов, уточните выбор",
            )

        return self._store(chosen)

    def add_show_by_id(self, tvmaze_id: int) -> AddResult:
        return self._store(self.tvmaze.get_show(tvmaze_id))

    def refresh_show(self, tvmaze_id: int) -> AddResult:
        """Перечитать сериал и его эпизоды из TVmaze."""
        stored = self.db.get_show_by_tvmaze_id(tvmaze_id)
        if stored is None:
            return AddResult(status="not_found", message=f"Сериал {tvmaze_id} не отслеживается")
        return self._store(self.tvmaze.get_show(tvmaze_id))

    def remove_show(self, tvmaze_id: int) -> bool:
        stored = self.db.get_show_by_tvmaze_id(tvmaze_id)
        if stored is None:
            return False
        self.db.remove_show(stored.id)
        return True

    def list_shows(self, *, now: datetime | None = None) -> list[TrackedShow]:
        return tracked_shows(self.db, now=now)

    def poll(
        self,
        *,
        now: datetime | None = None,
        lookback_hours: float | None = None,
        channel: str = NOTIFY_CHANNEL,
    ) -> PollReport:
        """Опросить расписание и собрать, что положено сообщить.

        Список эпизодов каждый раз берётся без кэша: у `/shows/{id}/episodes`
        нет поля `updated`, и единственный способ узнать о переносе эфира —
        спросить. Лимит запросов держит RateLimiter, а не этот код.
        """
        moment = now if now is not None else utcnow()
        hours = self.config.notify_lookback_hours if lookback_hours is None else lookback_hours
        since = moment - timedelta(hours=hours) if hours > 0 else None

        polls: list[ShowPoll] = []
        for show in self.db.list_shows():
            polls.append(self._poll_show(show))

        known = {show.id: show for show in self.db.list_shows()}
        due = tuple(
            DueEpisode(show=known[row_id], episode=episode)
            for row_id, episode in self.db.due_episodes(
                now=moment, since=since, channel=channel
            )
            if row_id in known
        )
        return PollReport(checked_at=moment, shows=tuple(polls), due=due)

    def _poll_show(self, show: Show) -> ShowPoll:
        try:
            fresh = self.tvmaze.get_show(show.tvmaze_id)
            episodes = self.tvmaze.get_episodes(show.tvmaze_id, refresh=True)
        except TvMazeError as exc:
            return ShowPoll(
                show=show,
                episodes_total=self.db.count_episodes(show.id),
                new_episodes=0,
                updated_episodes=0,
                without_time=self.db.count_episodes_without_time(show.id),
                error=str(exc),
            )
        stored = self.db.upsert_show(fresh)
        added, updated = self.db.upsert_episodes(stored.id, episodes)
        return ShowPoll(
            show=stored,
            episodes_total=self.db.count_episodes(stored.id),
            new_episodes=added,
            updated_episodes=updated,
            without_time=self.db.count_episodes_without_time(stored.id),
        )

    def _store(self, show: TvMazeShow) -> AddResult:
        known = self.db.get_show_by_tvmaze_id(show.id) is not None
        stored = self.db.upsert_show(show)
        episodes = self.tvmaze.get_episodes(show.id)
        added, updated = self.db.upsert_episodes(stored.id, episodes)
        message = f"эпизодов в базе: {len(episodes)} (новых {added}, обновлено {updated})"
        status = "exists" if known else "added"
        return AddResult(status=status, show=stored, episodes=len(episodes), message=message)