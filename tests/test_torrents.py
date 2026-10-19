"""Поиск раздач: повторы, отбор находок и отправка. Всё на фейковом поиске, без сети."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pytest

from series_watcher.config import Config
from series_watcher.db import Database
from series_watcher.finders import FinderError, TorrentHit
from series_watcher.finders.torrent_search import _NOISE, _fits, _to_hit, TorrentSearchFinder
from series_watcher.models import EpisodeRecord
from series_watcher.schedule import TvMazeSchedule
from series_watcher.service import SeriesWatcher
from series_watcher.tvmaze import parse_show

from conftest import load_fixture

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


@dataclass
class FakeFinder:
    """Поиск без сети: что в сценарии теста — то и найдёт."""

    name: str = "fake"
    description: str = "фейковый поиск"
    is_available: bool = True
    script: dict[str, Sequence[TorrentHit]] = field(default_factory=dict)
    searched: list[str] = field(default_factory=list)
    magnets: dict[str, str] = field(default_factory=dict)
    error: Exception | None = None

    def available(self) -> bool:
        return self.is_available

    def search(self, show: str, season: int, number: int, *, limit: int = 5) -> Sequence[TorrentHit]:
        if self.error is not None:
            raise self.error
        self.searched.append(f"{show} s{season:02d}e{number:02d}")
        return list(self.script.get(f"{show.lower()} s{season:02d}e{number:02d}", ()))[:limit]

    def magnet(self, torrent_id: str) -> str | None:
        # Явное None в magnets значит «трекер не отдал».
        if torrent_id in self.magnets:
            return self.magnets[torrent_id]
        return f"magnet:?xt=urn:btih:{torrent_id}"


def hit(filename: str, *, seeders: int = 5, source: str = "apibay", size: str = "1 GiB"):
    return TorrentHit(
        torrent_id=filename.lower().replace(" ", "-"),
        filename=filename,
        source=source,
        seeders=seeders,
        leechers=1,
        size=size,
        page_url=f"https://example.tv/{filename}",
    )


@pytest.fixture
def parts(tmp_path):
    db = Database(tmp_path / "torrents.db")
    show = db.upsert_show("tvmaze", TvMazeSchedule(None)._show(parse_show(load_fixture("show"))))
    db.touch_show(show.id)
    db.upsert_episodes(
        show.id,
        [EpisodeRecord(season=4, number=1, name="Identity Theft", aired_at=NOW - timedelta(hours=2))],
    )
    yield db, show
    db.close()


def watcher(db: Database, finder, *, config=None) -> SeriesWatcher:
    return SeriesWatcher(
        config=config or Config(torrent_search_enabled=True, notify_quiet_hours=None),
        db=db,
        source=None,
        notifier=None,
        finder=finder,
    )


def due(db: Database):
    show_id, episode = db.episodes_to_search(
        now=NOW, channel="telegram", retry_hours=6, retry_days=3
    )[0]
    return show_id, episode


def test_found_hits_are_saved_and_reported(parts):
    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01 1080p WEB-DL", seeders=42)]})
    w = watcher(db, finder)
    _, episode = due(db)

    result = w.search_torrents(type("Due", (), {"show": show, "episode": episode}))

    assert result.found
    rows = db.torrents_for(episode.id)
    assert [row.filename for row in rows] == ["Severance S04E01 1080p WEB-DL"]
    assert rows[0].seeders == 42
    message = w.torrent_message(show, episode)
    assert "Severance S04E01 1080p WEB-DL" in message
    assert "сидов 42" in message
    assert "на ваше усмотрение" in message


def test_dead_torrents_are_not_saved(parts):
    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=0)]})
    w = watcher(db, finder)
    _, episode = due(db)

    result = w.search_torrents(type("Due", (), {"show": show, "episode": episode}))

    assert not result.found
    assert result.detail == "пока ничего живого"
    assert db.torrents_for(episode.id) == []
    # Поиск был: пауза до следующей попытки отсчитывается от него.
    assert db.last_torrent_search(episode.id) is not None


def test_retry_honours_pause_and_window(parts):
    db, show = parts
    finder = FakeFinder(script={"severance s04e02": [hit("Severance S04E02", seeders=3)]})
    w = watcher(db, finder)

    w.hunt_torrents(now=NOW)
    assert len(finder.searched) == 1

    # Раньше шести часов не ищем.
    assert w.hunt_torrents(now=NOW + timedelta(hours=1)) == []
    assert len(finder.searched) == 1

    # Следующая серия из окна ищется отдельно, старше трёх дней — нет.
    db.upsert_episodes(
        show.id,
        [EpisodeRecord(season=4, number=2, name=None, aired_at=NOW + timedelta(hours=7))],
    )
    found = w.hunt_torrents(now=NOW + timedelta(hours=8))
    # Новая серия ищется сразу; старая — потому что прошла её шестичасовая пауза.
    assert [item.episode.number for item in found] == [2, 1]
    assert finder.searched.count("Severance s04e02") == 1

    db.upsert_episodes(
        show.id,
        [EpisodeRecord(season=4, number=3, name=None, aired_at=NOW - timedelta(days=10))],
    )
    assert w.hunt_torrents(now=NOW + timedelta(hours=9)) == []


def test_search_failure_is_no_crash(parts):
    db, show = parts
    w = watcher(db, FakeFinder(error=FinderError("трекер недоступен")))
    _, episode = due(db)

    result = w.search_torrents(type("Due", (), {"show": show, "episode": episode}))

    assert not result.found
    assert "трекер недоступен" in result.detail


def test_gone_finder_is_reported_plainly(parts):
    db, show = parts
    w = watcher(db, FakeFinder(is_available=False))
    _, episode = due(db)

    result = w.search_torrents(type("Due", (), {"show": show, "episode": episode}))

    assert not result.found
    assert "pip install torrent-search-mcp" in result.detail


@dataclass(frozen=True)
class FakeSource:
    """Источник без сети: тот же сериал и эпизоды, что уже в базе."""

    name: str = "tvmaze"
    record: object = None
    episodes: tuple = ()

    def get_show(self, external_id, *, fresh: bool = False):
        return self.record

    def get_episodes(self, external_id, *, fresh: bool = False):
        return self.episodes

    def get_schedule(self, day=None):
        return ()


def test_run_sends_torrent_after_episode_notice(tmp_path):
    """Сначала «вышел», потом «раздача нашлась» — и каждое ровно один раз."""
    from series_watcher.notify import Notifier, SendResult

    db = Database(tmp_path / "run.db")
    record = TvMazeSchedule(None)._show(parse_show(load_fixture("show")))
    episode_records = (
        EpisodeRecord(season=4, number=1, name=None, aired_at=NOW),
    )
    notifier = Notifier(Config(notify_quiet_hours=None), db, transport=object())
    notifier.send = lambda text, *, html=False: SendResult(sent=True)  # type: ignore[method-assign]
    show = db.upsert_show("tvmaze", record)
    db.touch_show(show.id)
    db.upsert_episodes(show.id, episode_records)
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=7)]})
    w = SeriesWatcher(
        config=Config(torrent_search_enabled=True, notify_quiet_hours=None),
        db=db,
        source=FakeSource(record=record, episodes=episode_records),
        notifier=notifier,
        finder=finder,
    )

    report = w.run(now=NOW)

    assert report.sent == 1
    assert report.found_torrents == 1
    kinds = [row["kind"] for row in db._conn.execute("SELECT kind FROM notifications ORDER BY id")]
    assert kinds == ["new_episode", "torrent"]

    report = w.run(now=NOW + timedelta(hours=1))
    assert report.sent == 0
    assert report.found_torrents == 0


def test_hunt_disabled_means_nothing_searched(parts):
    db, _ = parts
    finder = FakeFinder()
    w = SeriesWatcher(
        config=Config(torrent_search_enabled=False, notify_quiet_hours=None),
        db=db,
        source=None,
        notifier=None,
        finder=finder,
    )

    assert w.hunt_torrents(now=NOW) == []
    assert finder.searched == []


def test_finder_filters_and_ranks():
    """Мусор отсекается: игра — прочь, чужая серия — прочь, пакет — прочь."""
    finder = TorrentSearchFinder()

    class Item:
        def __init__(self, filename, seeders=5):
            self.filename = filename
            self.seeders = seeders
            self.leechers = 1
            self.size = "1 GiB"
            self.source = "apibay"
            self.id = "x"
            self.page_url = None

    right = _to_hit(Item("Slow Horses S02E05 1080p"))
    other = _to_hit(Item("Slow Horses S02E06 1080p"))
    pack = _to_hit(Item("Slow Horses S02E05-08 1080p"))
    game = _to_hit(Item("Some-game-cpy-repack"))
    no_code = _to_hit(Item("Just a show name"))

    assert right is not None and _fits(right, 2, 5)
    assert other is not None and not _fits(other, 2, 5)
    assert pack is not None and not _fits(pack, 2, 5)
    assert game is None and _NOISE.search("Some-game-cpy-repack") is not None
    assert no_code is None or not _fits(no_code, 2, 5)


def test_finder_query_is_lowercase_with_code():
    """Запрос — нижний регистр и s02e05: так понимают трекеры."""

    class Spy(FakeFinder):
        def search(self, show, season, number, *, limit=5):
            FakeFinder.search(self, show, season, number, limit=limit)
            return []

    spy = Spy()
    spy.search("Slow Horses", 2, 5)
    assert spy.searched == ["Slow Horses s02e05"]


def test_where_torrent_intent_shows_stored_hits(parts):
    """«Где скачать S04E01»: если раздачи уже найдены, отдаём их из базы."""
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01 1080p", seeders=9)]})
    w = watcher(db, finder)
    _, episode = due(db)
    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)
    assert finder.searched == ["Severance s04e01"]  # первый поиск

    reply = w.handle(rules("где скачать S04E01 Severance"))

    assert "Severance S04E01 1080p" in reply.text
    assert finder.searched == ["Severance s04e01"]  # повторно не ходили в сеть


def test_where_torrent_intent_searches_when_nothing_known(parts):
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=4)]})
    w = watcher(db, finder)
    _, episode = due(db)
    db.mark_torrent_search(episode.id, hits=0, at=NOW)  # искали, пусто

    reply = w.handle(rules("где скачать S04E01 Severance"))

    assert "Severance S04E01" in reply.text
    assert finder.searched == ["Severance s04e01"]


def test_where_torrent_says_nothing_alive(parts):
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=0)]})
    w = watcher(db, finder)
    reply = w.handle(rules("где скачать S04E01 Severance"))

    assert "пока нет" in reply.text


def test_where_torrent_asks_which_show(parts):
    from series_watcher.nlu import rules

    db, _ = parts
    w = watcher(db, FakeFinder())
    reply = w.handle(rules("где скачать"))

    assert "Назовите сериал и серию" in reply.text


def test_where_torrent_without_package_says_so(parts):
    from series_watcher.nlu import rules

    db, _ = parts
    w = watcher(db, FakeFinder(is_available=False))
    reply = w.handle(rules("где скачать S04E01 Severance"))

    assert "pip install torrent-search-mcp" in reply.text


def test_magnet_is_stored_and_shown(parts):
    """Магнет дозапрашивается для показываемой раздачи и переживает перезапуск."""
    db, show = parts
    finder = FakeFinder(
        script={"severance s04e01": [hit("Severance S04E01 1080p", seeders=11)]}
    )
    finder.magnets = {"severance-s04e01-1080p": "magnet:?xt=urn:btih:AAA"}
    w = watcher(db, finder)
    _, episode = due(db)

    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)

    rows = db.torrents_for(episode.id)
    assert rows[0].magnet == "magnet:?xt=urn:btih:AAA"
    message = w.torrent_message(show, episode)
    assert "magnet" in message


def test_missing_magnet_does_not_break_the_message(parts):
    """Трекер мог не отдать магнет: показываем страницу и не падаем."""
    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=5)]})
    finder.magnets = {"severance-s04e01": None}  # трекер не отдал магнет
    w = watcher(db, finder)
    _, episode = due(db)

    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)

    message = w.torrent_message(show, episode)
    assert "Severance S04E01" in message
    assert "magnet" not in message


def test_torrent_message_is_escaped(parts):
    """Имя файла с `&` и `<` не должно ломать разметку Telegram."""
    from series_watcher.notify import esc

    db, show = parts
    tricky = "Severance S04E01 1080p <b>&</b> WEB-DL"
    finder = FakeFinder(script={"severance s04e01": [hit(tricky, seeders=3)]})
    w = watcher(db, finder)
    _, episode = due(db)
    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)

    message = w.torrent_message(show, episode)
    assert "&amp;" in message
    assert "<b>&</b>" not in message
    assert esc("<b>") == "&lt;b&gt;"


def _list_intent():
    from series_watcher.nlu import rules

    return rules("список")


def test_torrent_message_has_no_tvmaze_credit(parts):
    """В ответе о раздачах нет данных TVmaze, значит нет и атрибуции.

    Описание, расписание и каталог тут не показываются: код серии человек назвал
    сам, раздачи пришли с трекеров. Подпись TVmaze здесь была бы формальностью.
    """
    from series_watcher.notify import TVMAZE_CREDIT

    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=5)]})
    w = watcher(db, finder)
    _, episode = due(db)
    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)

    message = w.torrent_message(show, episode)

    assert TVMAZE_CREDIT not in message
    assert "tvmaze.com" not in message
    # А ответ про отслеживаемые сериалы атрибуцию по-прежнему несёт.
    assert TVMAZE_CREDIT in w.handle(_list_intent()).text


def test_unknown_episode_says_so_instead_of_a_neighbour(parts):
    """MobLand отслеживается с S02E05. Спросили S02E04 — отвечаем «нет такой»,
    а не подсовываем соседнюю серию: человек-то просил конкретную."""
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder(script={"severance s04e02": [hit("Severance S04E02", seeders=9)]})
    w = watcher(db, finder)

    reply = w.handle(rules("найди раздачи для Severance S04E02"))

    assert "нет серии S04E02" in reply.text
    assert finder.searched == []


def test_magnet_link_is_pruned_to_hash_and_name():
    """Трекеры в магнете раздувают ссылку и делают её нечитаемой: оставляем хеш."""
    from series_watcher.finders import prune_magnet as _prune

    long = (
        "magnet:?xt=urn:btih:0a9abd14069a0608e7af5545682be8269b42bec3"
        "&dn=MobLand+S02E04&tr=udp%3A%2F%2Fa&tr=udp%3A%2F%2Fb&tr=http%3A%2F%2Fc"
    )
    assert _prune(long) == (
        "magnet:?xt=urn:btih:0a9abd14069a0608e7af5545682be8269b42bec3&dn=MobLand+S02E04"
    )
    assert len(_prune(long)) < len(long)
    assert _prune(None) is None


def test_on_demand_search_asks_for_magnets(parts):
    """Путь «по запросу» тоже тянет магнеты — раньше тянул только фоновый."""
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=8)]})
    finder.magnets = {"severance-s04e01": "magnet:?xt=urn:btih:DEADBEEF"}
    w = watcher(db, finder)

    reply = w.handle(rules("найди раздачи для Severance S04E01"))

    assert "magnet" in reply.text
    assert db.torrents_for(due(db)[1].id)[0].magnet == "magnet:?xt=urn:btih:DEADBEEF"


def test_cached_rows_get_missing_magnets_filled_in(parts):
    """Раздача в базе без магнета: при показе дозапрашиваем и дописываем."""
    from series_watcher.nlu import rules

    db, show = parts
    finder = FakeFinder()
    finder.magnets = {"severance-s04e01": "magnet:?xt=urn:btih:CAFEBABE"}
    w = watcher(db, finder)
    _, episode = due(db)
    db.save_torrents(episode.id, [hit("Severance S04E01", seeders=4)])  # без магнета

    reply = w.handle(rules("найди раздачи для Severance S04E01"))

    assert "magnet" in reply.text
    assert db.torrents_for(episode.id)[0].magnet == "magnet:?xt=urn:btih:CAFEBABE"


def test_magnet_is_text_not_a_link(parts):
    """Telegram не делает `magnet:` кликабельным: ссылка молча показывалась как
    слово «magnet» без единого действия. Поэтому магнет — моноширинный текст."""
    db, show = parts
    finder = FakeFinder(script={"severance s04e01": [hit("Severance S04E01", seeders=8)]})
    finder.magnets = {"severance-s04e01": "magnet:?xt=urn:btih:DEADBEEF"}
    w = watcher(db, finder)
    _, episode = due(db)
    w.search_torrents(type("Due", (), {"show": show, "episode": episode}), now=NOW)

    message = w.torrent_message(show, episode)

    assert "magnet:?xt=urn:btih:DEADBEEF" in message
    assert 'href="magnet:' not in message
    # А ссылка на трекер при этом осталась настоящей ссылкой.
    assert '<a href="https://example.tv/' in message
