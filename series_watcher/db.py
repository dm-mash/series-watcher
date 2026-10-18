"""Память сервиса: сериалы, эпизоды, отправки, отказы.

Хранится SQLite. Время эфира лежит в UTC в виде ISO-строки, поэтому сравнение
и сортировка в SQL работают как с обычным текстом. Ключ эпизода —
`(show_id, season, number)`: перенос эфира меняет дату, но не эпизод.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS shows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tvmaze_id       INTEGER NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    kind            TEXT,
    status          TEXT,
    network         TEXT,
    premiered       TEXT,
    ended           TEXT,
    runtime         INTEGER,
    language        TEXT,
    genres          TEXT    NOT NULL DEFAULT '[]',
    summary         TEXT,
    url             TEXT,
    image           TEXT,
    thetvdb_id      INTEGER,
    imdb_id         TEXT,
    rating          REAL,
    source_updated  INTEGER,
    added_at        TEXT    NOT NULL,
    checked_at      TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id            INTEGER NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    tvmaze_episode_id  INTEGER,
    season             INTEGER NOT NULL,
    number             INTEGER NOT NULL,
    name               TEXT,
    airstamp           TEXT,
    airdate            TEXT,
    airtime            TEXT,
    runtime            INTEGER,
    summary            TEXT,
    watched_at         TEXT,
    created_at         TEXT    NOT NULL,
    UNIQUE (show_id, season, number)
);

CREATE INDEX IF NOT EXISTS idx_episodes_show_airstamp
    ON episodes (show_id, airstamp);
CREATE INDEX IF NOT EXISTS idx_episodes_airstamp
    ON episodes (airstamp);

CREATE TABLE IF NOT EXISTS notifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    channel     TEXT    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'new_episode',
    sent_at     TEXT    NOT NULL,
    UNIQUE (episode_id, channel, kind)
);

CREATE TABLE IF NOT EXISTS rejects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT    NOT NULL CHECK (kind IN ('not_interested', 'already_watched')),
    tvmaze_id  INTEGER,
    title      TEXT,
    created_at TEXT    NOT NULL,
    UNIQUE (kind, tvmaze_id)
);

CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    payload    TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""

#: Канал один: сервис рассчитан на одного пользователя.
TELEGRAM_CHANNEL = "telegram"

#: Виды отказов от рекомендаций.
NOT_INTERESTED = "not_interested"
ALREADY_WATCHED = "already_watched"
REJECT_KINDS = (NOT_INTERESTED, ALREADY_WATCHED)

SHOW_COLUMNS = (
    "tvmaze_id, name, kind, status, network, premiered, ended, runtime, language, "
    "genres, summary, url, image, thetvdb_id, imdb_id, rating, source_updated, "
    "added_at, checked_at, id"
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return to_iso_utc(utcnow())


def to_iso_utc(value: datetime | None) -> str | None:
    """ISO-строка в UTC: единственный формат хранения времени эфира."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Show:
    id: int
    tvmaze_id: int
    name: str
    kind: str | None
    status: str | None
    network: str | None
    premiered: str | None
    ended: str | None
    runtime: int | None
    language: str | None
    genres: tuple[str, ...]
    summary: str | None
    url: str | None
    image: str | None
    thetvdb_id: int | None
    imdb_id: str | None
    rating: float | None
    source_updated: int | None
    added_at: str
    checked_at: str | None


@dataclass(frozen=True)
class Episode:
    id: int
    show_id: int
    season: int
    number: int
    name: str | None
    airstamp: datetime | None
    airdate: str | None
    airtime: str | None
    runtime: int | None
    summary: str | None
    watched_at: str | None
    tvmaze_episode_id: int | None

    @property
    def code(self) -> str:
        return f"S{self.season:02d}E{self.number:02d}" if self.season else f"E{self.number:02d}"


@dataclass(frozen=True)
class Reject:
    """Отказ от рекомендации: «не интересно» или «уже смотрел»."""

    kind: str
    tvmaze_id: int
    title: str | None
    created_at: str

    @property
    def label(self) -> str:
        return "уже смотрел" if self.kind == ALREADY_WATCHED else "не интересно"


@dataclass(frozen=True)
class TrackedShow:
    """Сериал из списка вместе с ближайшим невышедшим эпизодом."""

    show: Show
    next_episode: Episode | None
    episodes_total: int


class Database:
    def __init__(self, path: Path | str, *, clock=utcnow_iso) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            parent = Path(self.path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._clock = clock
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- сериалы ---------------------------------------------------------

    def upsert_show(self, show) -> Show:
        """Записать сериал по данным TVmaze, не трогая время добавления."""
        now = self._clock()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO shows (
                    tvmaze_id, name, kind, status, network, premiered, ended, runtime,
                    language, genres, summary, url, image, thetvdb_id, imdb_id, rating,
                    source_updated, added_at, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tvmaze_id) DO UPDATE SET
                    name = excluded.name,
                    kind = excluded.kind,
                    status = excluded.status,
                    network = excluded.network,
                    premiered = excluded.premiered,
                    ended = excluded.ended,
                    runtime = excluded.runtime,
                    language = excluded.language,
                    genres = excluded.genres,
                    summary = excluded.summary,
                    url = excluded.url,
                    image = excluded.image,
                    thetvdb_id = excluded.thetvdb_id,
                    imdb_id = excluded.imdb_id,
                    rating = excluded.rating,
                    source_updated = excluded.source_updated,
                    checked_at = excluded.checked_at
                """,
                (
                    show.id,
                    show.name,
                    show.kind,
                    show.status,
                    show.network,
                    show.premiered,
                    show.ended,
                    show.runtime,
                    show.language,
                    json.dumps(list(show.genres), ensure_ascii=False),
                    show.summary,
                    show.url,
                    show.image,
                    show.thetvdb_id,
                    show.imdb_id,
                    show.rating,
                    show.updated,
                    now,
                    now,
                ),
            )
        stored = self.get_show_by_tvmaze_id(show.id)
        assert stored is not None
        return stored

    def get_show(self, show_id: int) -> Show | None:
        row = self._conn.execute(
            f"SELECT {SHOW_COLUMNS} FROM shows WHERE id = ?", (show_id,)
        ).fetchone()
        return _row_to_show(row) if row else None

    def get_show_by_tvmaze_id(self, tvmaze_id: int) -> Show | None:
        row = self._conn.execute(
            f"SELECT {SHOW_COLUMNS} FROM shows WHERE tvmaze_id = ?", (tvmaze_id,)
        ).fetchone()
        return _row_to_show(row) if row else None

    def find_shows(self, query: str) -> list[Show]:
        """Отслеживаемые сериалы по названию: сначала точное совпадение."""
        from .tvmaze import normalize_name

        needle = normalize_name(query)
        if not needle:
            return []
        shows = self.list_shows()
        exact = [show for show in shows if normalize_name(show.name) == needle]
        if exact:
            return exact
        return [show for show in shows if needle in normalize_name(show.name)]

    def list_shows(self) -> list[Show]:
        rows = self._conn.execute(
            f"SELECT {SHOW_COLUMNS} FROM shows ORDER BY added_at, id"
        ).fetchall()
        return [_row_to_show(row) for row in rows]

    def touch_show(self, show_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE shows SET checked_at = ? WHERE id = ?", (self._clock(), show_id)
            )

    def remove_show(self, show_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM shows WHERE id = ?", (show_id,))

    # --- эпизоды --------------------------------------------------------

    def upsert_episodes(self, show_id: int, episodes: Iterable) -> tuple[int, int]:
        """Сложить эпизоды сериала. Возвращает (добавлено, обновлено).

        Эпизод опознаётся по `(show_id, season, number)`, поэтому перенос эфира
        обновляет ту же строку и не рождает дубликат.
        """
        now = self._clock()
        added = updated = 0
        with self._conn:
            for episode in episodes:
                known = self._conn.execute(
                    "SELECT 1 FROM episodes WHERE show_id = ? AND season = ? AND number = ?",
                    (show_id, episode.season, episode.number),
                ).fetchone()
                self._conn.execute(
                    """
                    INSERT INTO episodes (
                        show_id, tvmaze_episode_id, season, number, name, airstamp,
                        airdate, airtime, runtime, summary, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(show_id, season, number) DO UPDATE SET
                        tvmaze_episode_id = excluded.tvmaze_episode_id,
                        name = excluded.name,
                        airstamp = excluded.airstamp,
                        airdate = excluded.airdate,
                        airtime = excluded.airtime,
                        runtime = excluded.runtime,
                        summary = excluded.summary
                    """,
                    (
                        show_id,
                        episode.id,
                        episode.season,
                        episode.number,
                        episode.name,
                        to_iso_utc(episode.airstamp),
                        episode.airdate,
                        episode.airtime,
                        episode.runtime,
                        episode.summary,
                        now,
                    ),
                )
                if known is None:
                    added += 1
                else:
                    updated += 1
        return added, updated

    def list_episodes(self, show_id: int) -> list[Episode]:
        rows = self._conn.execute(
            """
            SELECT id, show_id, tvmaze_episode_id, season, number, name, airstamp,
                   airdate, airtime, runtime, summary, watched_at
            FROM episodes WHERE show_id = ? ORDER BY season, number
            """,
            (show_id,),
        ).fetchall()
        return [_row_to_episode(row) for row in rows]

    def next_episode(self, show_id: int, *, after: datetime | None = None) -> Episode | None:
        after_iso = to_iso_utc(after if after is not None else utcnow())
        row = self._conn.execute(
            """
            SELECT id, show_id, tvmaze_episode_id, season, number, name, airstamp,
                   airdate, airtime, runtime, summary, watched_at
            FROM episodes
            WHERE show_id = ? AND airstamp IS NOT NULL AND airstamp > ?
            ORDER BY airstamp LIMIT 1
            """,
            (show_id, after_iso),
        ).fetchone()
        return _row_to_episode(row) if row else None

    def count_episodes(self, show_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM episodes WHERE show_id = ?", (show_id,)
        ).fetchone()
        return int(row["n"])

    def count_episodes_without_time(self, show_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM episodes WHERE show_id = ? AND airstamp IS NULL", (show_id,)
        ).fetchone()
        return int(row["n"])

    def due_episodes(
        self,
        *,
        now: datetime,
        channel: str,
        since: datetime | None = None,
        show_row_id: int | None = None,
    ) -> list[tuple[int, Episode]]:
        """Эпизоды, которые уже вышли, но о которых ещё не сообщали.

        «Вышел» — значит `airstamp` в UTC не позже `now`. Эпизоды без времени
        сюда не попадают: единственный источник времени — `airstamp`.
        Окно `since` нужно, чтобы добавление сериала с историей не превратилось
        в поток из нескольких сотен уведомлений.

        Возвращает пары (строка сериала в `shows`, эпизод): сам сериал вызывающий
        код обычно уже держит в руках.
        """
        clauses = [
            "e.airstamp IS NOT NULL",
            "e.airstamp <= ?",
            "e.watched_at IS NULL",
            "NOT EXISTS (SELECT 1 FROM notifications n WHERE n.episode_id = e.id AND n.channel = ?)",
        ]
        params: list[object] = [to_iso_utc(now), channel]
        if since is not None:
            clauses.append("e.airstamp >= ?")
            params.append(to_iso_utc(since))
        if show_row_id is not None:
            clauses.append("e.show_id = ?")
            params.append(show_row_id)
        rows = self._conn.execute(
            f"""
            SELECT id, show_id, tvmaze_episode_id, season, number, name, airstamp,
                   airdate, airtime, runtime, summary, watched_at
            FROM episodes e
            WHERE {' AND '.join(clauses)}
            ORDER BY airstamp, show_id, season, number
            """,
            params,
        ).fetchall()
        return [(row["show_id"], _row_to_episode(row)) for row in rows]

    def upcoming_episodes(self, *, since: datetime, until: datetime) -> list[tuple[Show, Episode]]:
        """Что выходит между `since` и `until`, по времени эфира в UTC."""
        rows = self._conn.execute(
            """
            SELECT id, show_id, tvmaze_episode_id, season, number, name, airstamp,
                   airdate, airtime, runtime, summary, watched_at
            FROM episodes
            WHERE airstamp IS NOT NULL AND airstamp >= ? AND airstamp < ?
            ORDER BY airstamp, show_id, season, number
            """,
            (to_iso_utc(since), to_iso_utc(until)),
        ).fetchall()
        shows = {show.id: show for show in self.list_shows()}
        return [
            (shows[episode.show_id], episode)
            for episode in (_row_to_episode(row) for row in rows)
            if episode.show_id in shows
        ]

    # --- отправки -------------------------------------------------------

    def mark_watched(self, episode_id: int, *, watched: bool = True) -> None:
        """Отметка «я посмотрел»: про таком эпизоде больше не напоминаем."""
        with self._conn:
            self._conn.execute(
                "UPDATE episodes SET watched_at = ? WHERE id = ?",
                (self._clock() if watched else None, episode_id),
            )

    def is_sent(self, episode_id: int, channel: str, kind: str = "new_episode") -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM notifications WHERE episode_id = ? AND channel = ? AND kind = ?",
            (episode_id, channel, kind),
        ).fetchone()
        return row is not None

    def mark_sent(
        self, episode_id: int, channel: str, kind: str = "new_episode", *, sent_at: str | None = None
    ) -> bool:
        """Отметить отправку. `False`, если отметка уже была: повтор не шлём."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO notifications (episode_id, channel, kind, sent_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(episode_id, channel, kind) DO NOTHING
                """,
                (episode_id, channel, kind, sent_at or self._clock()),
            )
        return cursor.rowcount > 0

    # --- отказы ---------------------------------------------------------

    def reject(
        self, tvmaze_id: int, kind: str, *, title: str | None = None
    ) -> bool:
        """Отметить кандидата как неинтересного. Повтор молча игнорируется."""
        if kind not in REJECT_KINDS:
            raise ValueError(f"неизвестный вид отказа: {kind!r}")
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO rejects (kind, tvmaze_id, title, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(kind, tvmaze_id) DO NOTHING
                """,
                (kind, tvmaze_id, title, self._clock()),
            )
        return cursor.rowcount > 0

    def rejects(self) -> list[Reject]:
        rows = self._conn.execute(
            "SELECT kind, tvmaze_id, title, created_at FROM rejects ORDER BY created_at"
        ).fetchall()
        return [Reject(kind=row["kind"], tvmaze_id=row["tvmaze_id"], title=row["title"],
                       created_at=row["created_at"]) for row in rows]

    def rejected_ids(self) -> set[int]:
        return {row["tvmaze_id"] for row in self._conn.execute("SELECT tvmaze_id FROM rejects")}

    # --- HTTP-кэш -------------------------------------------------------

    def cache_get(self, key: str) -> tuple[str, str] | None:
        """Возвращает (payload, fetched_at) или None."""
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return row["payload"], row["fetched_at"]

    def cache_put(self, key: str, url: str, payload: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO http_cache (key, url, payload, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    url = excluded.url,
                    payload = excluded.payload,
                    fetched_at = excluded.fetched_at
                """,
                (key, url, payload, self._clock()),
            )

    def cache_clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM http_cache")


def _row_to_show(row: sqlite3.Row) -> Show:
    return Show(
        id=row["id"],
        tvmaze_id=row["tvmaze_id"],
        name=row["name"],
        kind=row["kind"],
        status=row["status"],
        network=row["network"],
        premiered=row["premiered"],
        ended=row["ended"],
        runtime=row["runtime"],
        language=row["language"],
        genres=tuple(json.loads(row["genres"] or "[]")),
        summary=row["summary"],
        url=row["url"],
        image=row["image"],
        thetvdb_id=row["thetvdb_id"],
        imdb_id=row["imdb_id"],
        rating=row["rating"],
        source_updated=row["source_updated"],
        added_at=row["added_at"],
        checked_at=row["checked_at"],
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    airstamp = row["airstamp"]
    return Episode(
        id=row["id"],
        show_id=row["show_id"],
        tvmaze_episode_id=row["tvmaze_episode_id"],
        season=row["season"],
        number=row["number"],
        name=row["name"],
        airstamp=datetime.fromisoformat(airstamp) if airstamp else None,
        airdate=row["airdate"],
        airtime=row["airtime"],
        runtime=row["runtime"],
        summary=row["summary"],
        watched_at=row["watched_at"],
    )


def tracked_shows(db: Database, *, now: datetime | None = None) -> list[TrackedShow]:
    """Список отслеживаемых сериалов с ближайшим невышедшим эпизодом."""
    moment = now if now is not None else utcnow()
    items: list[TrackedShow] = []
    for show in db.list_shows():
        next_episode = db.next_episode(show.id, after=moment)
        items.append(
            TrackedShow(
                show=show,
                next_episode=next_episode,
                episodes_total=db.count_episodes(show.id),
            )
        )
    items.sort(key=lambda item: (item.next_episode is None, _sort_key(item.next_episode)))
    return items


def _sort_key(episode: Episode | None) -> Sequence[float]:
    if episode is None or episode.airstamp is None:
        return (float("inf"),)
    return (episode.airstamp.timestamp(),)