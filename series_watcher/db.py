"""Память сервиса: сериалы, эпизоды, отправки, отказы.

Хранится SQLite. Время эфира лежит в UTC в виде ISO-строки, поэтому сравнение
и сортировка в SQL работают как с обычным текстом. Ключ эпизода —
`(show_id, season, number)`: перенос эфира меняет дату, но не эпизод.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS shows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
    rating          REAL,
    source_updated  INTEGER,
    added_at        TEXT    NOT NULL,
    checked_at      TEXT
);

CREATE TABLE IF NOT EXISTS identifiers (
    show_id     INTEGER NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    source      TEXT    NOT NULL,
    external_id TEXT    NOT NULL,
    PRIMARY KEY (source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_identifiers_show ON identifiers(show_id);

CREATE TABLE IF NOT EXISTS episodes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id            INTEGER NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    external_id        TEXT,
    season             INTEGER NOT NULL,
    number             INTEGER NOT NULL,
    name               TEXT,
    aired_at           TEXT,
    airtime            TEXT,
    runtime            INTEGER,
    summary            TEXT,
    summary_ru         TEXT,
    watched_at         TEXT,
    created_at         TEXT    NOT NULL,
    UNIQUE (show_id, season, number)
);

CREATE INDEX IF NOT EXISTS idx_episodes_show_aired
    ON episodes (show_id, aired_at);
CREATE INDEX IF NOT EXISTS idx_episodes_aired
    ON episodes (aired_at);

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
    source     TEXT    NOT NULL,
    external_id TEXT   NOT NULL,
    title      TEXT,
    created_at TEXT    NOT NULL,
    UNIQUE (kind, source, external_id)
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    payload    TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS torrents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    source      TEXT    NOT NULL,
    external_id TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    size        TEXT,
    seeders     INTEGER NOT NULL DEFAULT 0,
    leechers    INTEGER NOT NULL DEFAULT 0,
    url         TEXT,
    magnet      TEXT,
    found_at    TEXT    NOT NULL,
    UNIQUE (episode_id, source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_torrents_episode ON torrents (episode_id, seeders DESC);

-- Когда последний раз искали: по этому считаем паузу между попытками.
CREATE TABLE IF NOT EXISTS torrent_searches (
    episode_id  INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    searched_at TEXT    NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
"""

#: Канал один: сервис рассчитан на одного пользователя.
TELEGRAM_CHANNEL = "telegram"

#: Ключ настройки: чат, откуда бот последний раз получал сообщения.
CHAT_ID_SETTING = "telegram_chat_id"

#: Источник расписания по умолчанию: пока он один, но в базе ключ всегда с ним.
DEFAULT_SOURCE = "tvmaze"

#: Виды отказов от рекомендаций.
NOT_INTERESTED = "not_interested"
ALREADY_WATCHED = "already_watched"
REJECT_KINDS = (NOT_INTERESTED, ALREADY_WATCHED)

SHOW_COLUMNS = (
    "id, name, kind, status, network, premiered, ended, runtime, language, "
    "genres, summary, url, image, rating, source_updated, added_at, checked_at"
)


class SchemaError(RuntimeError):
    """Файл базы есть, а схема чужая — продолжать нельзя."""


def normalize_name(value: str) -> str:
    """Ключ сравнения названий: без регистра, артиклей и лишних пробелов."""
    text = " ".join((value or "").split()).casefold()
    for prefix in ("the ", "a ", "an "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip(" -:,!?")


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
class Identifier:
    """Сериал глазами одного источника: («tvmaze», «44933») или чужой базы."""

    source: str
    external_id: str

    @property
    def key(self) -> str:
        return f"{self.source}:{self.external_id}"


@dataclass(frozen=True)
class Show:
    id: int
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
    identifiers: tuple[Identifier, ...]
    rating: float | None
    source_updated: int | None
    added_at: str
    checked_at: str | None

    @property
    def key(self) -> Identifier | None:
        """Ключ активного источника — им пользуются ссылки, отказы и команды."""
        for identifier in self.identifiers:
            if identifier.source == DEFAULT_SOURCE:
                return identifier
        return self.identifiers[0] if self.identifiers else None

    @property
    def external_id(self) -> str | None:
        key = self.key
        return key.external_id if key else None

    @property
    def name_of(self) -> dict[str, str]:  # pragma: no cover - удобство отладки
        return {identifier.source: identifier.external_id for identifier in self.identifiers}


@dataclass(frozen=True)
class Episode:
    id: int
    show_id: int
    season: int
    number: int
    name: str | None
    aired_at: datetime | None
    runtime: int | None
    summary: str | None
    watched_at: str | None
    external_id: str | None = None
    summary_ru: str | None = None

    @property
    def code(self) -> str:
        return f"S{self.season:02d}E{self.number:02d}" if self.season else f"E{self.number:02d}"


@dataclass(frozen=True)
class TorrentRow:
    """Найденная раздача из базы. Магнет кэшируем здесь же: иначе пришлось бы
    дёргать трекер каждый раз, когда человек снова спросит про ту же серию."""

    source: str
    external_id: str
    filename: str
    size: str | None = None
    seeders: int = 0
    leechers: int = 0
    url: str | None = None
    magnet: str | None = None
    found_at: str | None = None


@dataclass(frozen=True)
class Reject:
    """Отказ от рекомендации: «не интересно» или «уже смотрел»."""

    kind: str
    identifier: Identifier
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
        self._check_schema()
        with self._conn:
            self._conn.executescript(SCHEMA)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Дописать колонки, которых нет в уже созданной таблице.

        `CREATE TABLE IF NOT EXISTS` молча оставляет старую таблицу как есть, а
        `SELECT magnet` на ней падает. Одна колонка — один `ALTER TABLE`, без
        пересборки базы.
        """
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(torrents)")}
        if columns and "magnet" not in columns:
            with self._conn:
                self._conn.execute("ALTER TABLE torrents ADD COLUMN magnet TEXT")

    def _check_schema(self) -> None:
        """Старая база не чинится молча: её удаляют и заводят заново."""
        tables = {
            row["name"]
            for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "shows" in tables and "identifiers" not in tables:
            raise SchemaError(
                f"{self.path}: база сделана до перехода на общую схему. "
                "Удалите файл и добавьте сериалы заново — миграции для старых баз нет."
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- сериалы ---------------------------------------------------------

    def upsert_show(self, source: str, record) -> Show:
        """Записать сериал из источника, не трогая время добавления."""
        now = self._clock()
        with self._conn:
            row = self._conn.execute(
                "SELECT show_id FROM identifiers WHERE source = ? AND external_id = ?",
                (source, record.external_id),
            ).fetchone()
            if row is not None:
                show_id = int(row["show_id"])
                self._conn.execute(
                    """
                    UPDATE shows SET
                        name = ?, kind = ?, status = ?, network = ?, premiered = ?,
                        ended = ?, runtime = ?, language = ?, genres = ?, summary = ?,
                        url = ?, image = ?, rating = ?, source_updated = ?, checked_at = ?
                    WHERE id = ?
                    """,
                    (
                        record.title,
                        record.kind,
                        record.status,
                        record.network,
                        record.premiered,
                        record.ended,
                        record.runtime,
                        record.language,
                        json.dumps(list(record.genres), ensure_ascii=False),
                        record.summary,
                        record.url,
                        record.image,
                        record.rating,
                        record.updated,
                        now,
                        show_id,
                    ),
                )
            else:
                cursor = self._conn.execute(
                    """
                    INSERT INTO shows (
                        name, kind, status, network, premiered, ended, runtime, language,
                        genres, summary, url, image, rating, source_updated, added_at, checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.title,
                        record.kind,
                        record.status,
                        record.network,
                        record.premiered,
                        record.ended,
                        record.runtime,
                        record.language,
                        json.dumps(list(record.genres), ensure_ascii=False),
                        record.summary,
                        record.url,
                        record.image,
                        record.rating,
                        record.updated,
                        now,
                        now,
                    ),
                )
                show_id = int(cursor.lastrowid)
            self.link_identifier(show_id, source, record.external_id)
            for name, value in (record.identifiers or {}).items():
                self.link_identifier(show_id, name, value)
        stored = self.get_show(show_id)
        assert stored is not None
        return stored

    def link_identifier(self, show_id: int, source: str, external_id: str) -> None:
        """Один сериал — много ключей: свой в источнике и в чужих базах."""
        self._conn.execute(
            "INSERT INTO identifiers (show_id, source, external_id) VALUES (?, ?, ?) "
            "ON CONFLICT(source, external_id) DO NOTHING",
            (show_id, source, str(external_id)),
        )

    def _with_identifiers(self, show: Show) -> Show:
        return replace(show, identifiers=self.identifiers_of(show.id))

    def identifiers_of(self, show_id: int) -> tuple[Identifier, ...]:
        rows = self._conn.execute(
            "SELECT source, external_id FROM identifiers WHERE show_id = ? ORDER BY source",
            (show_id,),
        ).fetchall()
        return tuple(Identifier(row["source"], row["external_id"]) for row in rows)

    def get_show(self, show_id: int) -> Show | None:
        row = self._conn.execute(
            f"SELECT {SHOW_COLUMNS} FROM shows WHERE id = ?", (show_id,)
        ).fetchone()
        return self._with_identifiers(_row_to_show(row)) if row else None

    def get_by_identifier(self, source: str, external_id: str | int) -> Show | None:
        """Сериал по ключу источника; голое число — это источник по умолчанию."""
        source, external_id = split_key(str(external_id), default_source=source)
        row = self._conn.execute(
            f"""
            SELECT {', '.join(f's.{name.strip()}' for name in SHOW_COLUMNS.split(','))}
            FROM identifiers i JOIN shows s ON s.id = i.show_id
            WHERE i.source = ? AND i.external_id = ?
            """,
            (source, external_id),
        ).fetchone()
        return self._with_identifiers(_row_to_show(row)) if row else None

    def find_by_name(self, source: str, external_id: str | int) -> Show | None:
        """Тот же поиск по названию — источник умеет искать, а не только по id."""
        return self.get_by_identifier(source, external_id)

    def find_shows(self, query: str) -> list[Show]:
        """Отслеживаемые сериалы по названию: сначала точное совпадение."""
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
        return [self._with_identifiers(_row_to_show(row)) for row in rows]

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
                        show_id, external_id, season, number, name, aired_at,
                        runtime, summary, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(show_id, season, number) DO UPDATE SET
                        external_id = excluded.external_id,
                        name = excluded.name,
                        aired_at = excluded.aired_at,
                        runtime = excluded.runtime,
                        summary = excluded.summary
                    """,
                    (
                        show_id,
                        episode.external_id,
                        episode.season,
                        episode.number,
                        episode.name,
                        to_iso_utc(episode.aired_at),
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
            SELECT id, show_id, external_id, season, number, name, aired_at,
                   runtime, summary, summary_ru, watched_at
            FROM episodes WHERE show_id = ? ORDER BY season, number
            """,
            (show_id,),
        ).fetchall()
        return [_row_to_episode(row) for row in rows]

    def next_episode(self, show_id: int, *, after: datetime | None = None) -> Episode | None:
        after_iso = to_iso_utc(after if after is not None else utcnow())
        row = self._conn.execute(
            """
            SELECT id, show_id, external_id, season, number, name, aired_at,
                   runtime, summary, summary_ru, watched_at
            FROM episodes
            WHERE show_id = ? AND aired_at IS NOT NULL AND aired_at > ?
            ORDER BY aired_at LIMIT 1
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
            "SELECT COUNT(*) AS n FROM episodes WHERE show_id = ? AND aired_at IS NULL", (show_id,)
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

        «Вышел» — значит `aired_at` в UTC не позже `now`. Эпизоды без времени
        сюда не попадают: единственный источник времени эфира.
        Окно `since` нужно, чтобы добавление сериала с историей не превратилось
        в поток из нескольких сотен уведомлений.

        Возвращает пары (строка сериала в `shows`, эпизод): сам сериал вызывающий
        код обычно уже держит в руках.
        """
        clauses = [
            "e.aired_at IS NOT NULL",
            "e.aired_at <= ?",
            "e.watched_at IS NULL",
            "NOT EXISTS (SELECT 1 FROM notifications n WHERE n.episode_id = e.id AND n.channel = ?)",
        ]
        params: list[object] = [to_iso_utc(now), channel]
        if since is not None:
            clauses.append("e.aired_at >= ?")
            params.append(to_iso_utc(since))
        if show_row_id is not None:
            clauses.append("e.show_id = ?")
            params.append(show_row_id)
        rows = self._conn.execute(
            f"""
            SELECT id, show_id, external_id, season, number, name, aired_at,
                   runtime, summary, summary_ru, watched_at
            FROM episodes e
            WHERE {' AND '.join(clauses)}
            ORDER BY aired_at, show_id, season, number
            """,
            params,
        ).fetchall()
        return [(row["show_id"], _row_to_episode(row)) for row in rows]

    def upcoming_episodes(self, *, since: datetime, until: datetime) -> list[tuple[Show, Episode]]:
        """Что выходит между `since` и `until`, по времени эфира в UTC."""
        rows = self._conn.execute(
            """
            SELECT id, show_id, external_id, season, number, name, aired_at,
                   runtime, summary, summary_ru, watched_at
            FROM episodes
            WHERE aired_at IS NOT NULL AND aired_at >= ? AND aired_at < ?
            ORDER BY aired_at, show_id, season, number
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

    def get_summary_ru(self, episode_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT summary_ru FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return row["summary_ru"] if row else None

    def set_summary_ru(self, episode_id: int, text: str | None) -> None:
        """Перевод и сжатие описания: один раз сделано — лежит в базе."""
        with self._conn:
            self._conn.execute(
                "UPDATE episodes SET summary_ru = ? WHERE id = ?", (text, episode_id)
            )

    def mark_watched(self, episode_id: int, *, watched: bool = True) -> None:
        """Отметка «я посмотрел»: про таком эпизоде больше не напоминаем."""
        with self._conn:
            self._conn.execute(
                "UPDATE episodes SET watched_at = ? WHERE id = ?",
                (self._clock() if watched else None, episode_id),
            )

    # --- раздачи ---------------------------------------------------------

    def save_torrents(self, episode_id: int, hits: Iterable) -> int:
        """Положить найденные раздачи. Повторный поиск не плодит дубликаты.

        Раздачи с нулевыми сидами не сохраняем: они мёртвые, и завтра их всё
        равно придётся искать заново.
        """
        now = self._clock()
        saved = 0
        with self._conn:
            for hit in hits:
                if getattr(hit, "seeders", 0) <= 0:
                    continue
                cursor = self._conn.execute(
                    """
                    INSERT INTO torrents (
                        episode_id, source, external_id, filename, size,
                        seeders, leechers, url, magnet, found_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(episode_id, source, external_id) DO UPDATE SET
                        seeders = excluded.seeders,
                        leechers = excluded.leechers,
                        url = excluded.url,
                        magnet = excluded.magnet
                    """,
                    (
                        episode_id,
                        hit.source,
                        hit.torrent_id,
                        hit.filename,
                        getattr(hit, "size", None),
                        int(getattr(hit, "seeders", 0) or 0),
                        int(getattr(hit, "leechers", 0) or 0),
                        getattr(hit, "page_url", None),
                        getattr(hit, "magnet", None),
                        now,
                    ),
                )
                saved += max(cursor.rowcount, 0)
        return saved

    def torrents_for(self, episode_id: int, *, limit: int = 5) -> list[TorrentRow]:
        """Найденные раздачи серии, сначала живые и с наибольшим числом сидов."""
        rows = self._conn.execute(
            """
            SELECT source, external_id, filename, size, seeders, leechers, url, magnet, found_at
            FROM torrents
            WHERE episode_id = ?
            ORDER BY seeders DESC, filename
            LIMIT ?
            """,
            (episode_id, limit),
        ).fetchall()
        return [TorrentRow(**dict(row)) for row in rows]

    def set_torrent_magnet(self, episode_id: int, row: "TorrentRow", magnet: str) -> None:
        """Дописать магнет к сохранённой раздаче.

        Отдельным запросом, а не перезаписью: магнет мог не отдаться в первый раз
        (трекер прячет), и обновлять надо только его одно поле. Серия в условии
        обязательна: одна и та же раздача может лежать в нескольких сериях.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE torrents SET magnet = ?
                WHERE episode_id = ? AND source = ? AND external_id = ?
                """,
                (magnet, episode_id, row.source, row.external_id),
            )

    def has_torrents(self, episode_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM torrents WHERE episode_id = ? LIMIT 1", (episode_id,)
        ).fetchone()
        return row is not None

    def mark_torrent_search(self, episode_id: int, *, hits: int = 0, at: datetime | None = None) -> None:
        """Запомнить, что поиск был: иначе следующий опрос полезет снова.

        Время передаётся вызывающим: у опроса может быть своё «сейчас», и пауза
        между попытками обязана считаться по нему, а не по настенным часам.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO torrent_searches (episode_id, searched_at, hits)
                VALUES (?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    searched_at = excluded.searched_at,
                    hits = excluded.hits
                """,
                (episode_id, to_iso_utc(at) if at else self._clock(), hits),
            )

    def last_torrent_search(self, episode_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT searched_at FROM torrent_searches WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return row["searched_at"] if row else None

    def episodes_to_search(
        self, *, now: datetime, channel: str, retry_hours: float, retry_days: float
    ) -> list[tuple[int, Episode]]:
        """Серии, которые пора поискать: вышли недавно, о раздаче не сообщали,
        а с прошлого поиска прошло достаточно времени.

        Окно в `retry_days` нужно потому, что раздачи появляются с задержкой:
        сразу после эфира их обычно ещё нет. Пауза в `retry_hours` бережёт
        трекеры: один поиск — это скрейпинг восьми сайтов. О находке сообщаем
        отдельно от «вышел»: одно не отменяет другое.
        """
        rows = self._conn.execute(
            """
            SELECT e.id, e.show_id, e.external_id, e.season, e.number, e.name, e.aired_at,
                   e.runtime, e.summary, e.summary_ru, e.watched_at
            FROM episodes e
            JOIN shows s ON s.id = e.show_id
            WHERE e.aired_at IS NOT NULL
              AND s.checked_at IS NOT NULL
              AND e.aired_at <= ?
              AND e.aired_at >= ?
              AND e.watched_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM notifications n
                  WHERE n.episode_id = e.id AND n.channel = ? AND n.kind = 'torrent'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM torrent_searches ts
                  WHERE ts.episode_id = e.id AND ts.searched_at >= ?
              )
            ORDER BY e.aired_at DESC, e.show_id, e.season, e.number
            """,
            (
                to_iso_utc(now),
                to_iso_utc(now - timedelta(days=retry_days)),
                channel,
                to_iso_utc(now - timedelta(hours=retry_hours)),
            ),
        ).fetchall()
        return [(row["show_id"], _row_to_episode(row)) for row in rows]

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
        self, external_id: str | int, kind: str, *, source: str = DEFAULT_SOURCE, title: str | None = None
    ) -> bool:
        """Отметить кандидата как неинтересного. Повтор молча игнорируется."""
        if kind not in REJECT_KINDS:
            raise ValueError(f"неизвестный вид отказа: {kind!r}")
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO rejects (kind, source, external_id, title, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(kind, source, external_id) DO NOTHING
                """,
                (kind, *split_key(str(external_id), default_source=source), title, self._clock()),
            )
        return cursor.rowcount > 0

    def rejects(self) -> list[Reject]:
        rows = self._conn.execute(
            "SELECT kind, source, external_id, title, created_at FROM rejects ORDER BY created_at"
        ).fetchall()
        return [
            Reject(
                kind=row["kind"],
                identifier=Identifier(row["source"], row["external_id"]),
                title=row["title"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def rejected_ids(self) -> set[str]:
        """Ключи вида «tvmaze:44933» — источник и идентификатор вместе."""
        rows = self._conn.execute("SELECT source, external_id FROM rejects").fetchall()
        return {f"{row['source']}:{row['external_id']}" for row in rows}

    # --- настройки -------------------------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, self._clock()),
            )

    def get_setting(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

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
        identifiers=(),
        rating=row["rating"],
        source_updated=row["source_updated"],
        added_at=row["added_at"],
        checked_at=row["checked_at"],
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    aired = row["aired_at"]
    return Episode(
        id=row["id"],
        show_id=row["show_id"],
        season=row["season"],
        number=row["number"],
        name=row["name"],
        aired_at=datetime.fromisoformat(aired) if aired else None,
        runtime=row["runtime"],
        summary=row["summary"],
        watched_at=row["watched_at"],
        external_id=row["external_id"],
        summary_ru=row["summary_ru"] if "summary_ru" in row.keys() else None,
    )


def split_key(key: str, *, default_source: str = DEFAULT_SOURCE) -> tuple[str, str]:
    """«tvmaze:44933» или просто «44933» — в виде источника и идентификатора."""
    if ":" in key:
        source, _, external_id = key.partition(":")
        return source.strip(), external_id.strip()
    return default_source, key.strip()


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
    if episode is None or episode.aired_at is None:
        return (float("inf"),)
    return (episode.aired_at.timestamp(),)