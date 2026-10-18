"""База: ключ эпизода, UTC в airstamp, порядок в списке, отметки об отправке."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from series_watcher.db import Database, tracked_shows

from conftest import load_fixture, store_show

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def episode(season: int, number: int, aired_at: datetime | None, name: str = "эпизод"):
    from series_watcher.models import EpisodeRecord

    return EpisodeRecord(season=season, number=number, name=name, aired_at=aired_at)


def add_show(db: Database, tvmaze_id: int, name: str):
    payload = dict(load_fixture("show"))
    payload["id"] = tvmaze_id
    payload["name"] = name
    return store_show(db, payload)


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def test_episodes_are_keyed_by_season_and_number(db):
    show = add_show(db, 1, "Первый")
    added, updated = db.upsert_episodes(show.id, [episode(1, 1, NOW), episode(1, 2, NOW)])
    assert (added, updated) == (2, 0)

    again, updated = db.upsert_episodes(show.id, [episode(1, 1, NOW)])
    assert (again, updated) == (0, 1)
    assert db.count_episodes(show.id) == 2


def test_reschedule_updates_the_same_episode_row(db):
    """Перенос эфира меняет дату, но не эпизод: новой строки быть не должно."""
    show = add_show(db, 1, "Первый")
    db.upsert_episodes(show.id, [episode(2, 5, NOW, "было")])
    db.upsert_episodes(show.id, [episode(2, 5, NOW + timedelta(days=7), "стало")])

    episodes = db.list_episodes(show.id)
    assert len(episodes) == 1
    assert episodes[0].aired_at == NOW + timedelta(days=7)
    assert episodes[0].name == "стало"


def test_sent_mark_follows_the_episode_not_the_air_time(db):
    """Отметка об отправке не должна слететь после переноса эфира."""
    show = add_show(db, 1, "Первый")
    db.upsert_episodes(show.id, [episode(1, 1, NOW)])
    stored = db.list_episodes(show.id)[0]

    assert db.mark_sent(stored.id, "telegram") is True
    assert db.is_sent(stored.id, "telegram") is True

    db.upsert_episodes(show.id, [episode(1, 1, NOW + timedelta(days=1))])
    moved = db.list_episodes(show.id)[0]
    assert moved.id == stored.id
    assert db.is_sent(moved.id, "telegram") is True
    assert db.mark_sent(moved.id, "telegram") is False
    assert db.is_sent(moved.id, "email") is False


def test_airstamp_is_stored_in_utc(db):
    show = add_show(db, 1, "Первый")
    local_evening = datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc)
    db.upsert_episodes(show.id, [episode(1, 2, local_evening), episode(1, 1, NOW)])

    stored = {item.code: item for item in db.list_episodes(show.id)}
    assert stored["S01E02"].aired_at.isoformat() == "2026-10-19T14:00:00+00:00"
    assert stored["S01E02"].aired_at.tzinfo is timezone.utc
    # Порядок в базе — по номеру эпизода, ближайший по времени — отдельный вопрос.
    assert db.next_episode(show.id, after=NOW - timedelta(hours=1)).code == "S01E01"


def test_next_episode_is_the_closest_one_in_the_future(db):
    show = add_show(db, 1, "Первый")
    past = NOW - timedelta(days=1)
    db.upsert_episodes(
        show.id,
        [
            episode(1, 1, past),
            episode(1, 3, NOW + timedelta(days=3)),
            episode(1, 2, NOW + timedelta(days=1)),
        ],
    )
    assert db.next_episode(show.id, after=NOW).number == 2
    assert db.next_episode(show.id, after=NOW + timedelta(days=2)).number == 3
    assert db.next_episode(show.id, after=NOW + timedelta(days=5)) is None


def test_episode_without_air_time_is_not_next(db):
    show = add_show(db, 1, "Первый")
    db.upsert_episodes(show.id, [episode(1, 1, None)])
    assert db.next_episode(show.id, after=NOW) is None


def test_tracked_shows_sorted_by_next_episode_with_empty_last(db):
    running = add_show(db, 75194, "Бегущий")
    finished = add_show(db, 44933, "Завершённый")
    db.upsert_episodes(running.id, [episode(1, 1, NOW + timedelta(hours=6))])
    db.upsert_episodes(finished.id, [episode(1, 1, NOW - timedelta(days=30))])

    items = tracked_shows(db, now=NOW)
    assert [item.show.name for item in items] == ["Бегущий", "Завершённый"]
    assert items[0].next_episode.aired_at == NOW + timedelta(hours=6)
    assert items[1].next_episode is None
    assert items[0].episodes_total == 1


def test_show_can_be_removed_with_its_episodes(db):
    show = add_show(db, 1, "Первый")
    db.upsert_episodes(show.id, [episode(1, 1, NOW)])
    stored = db.list_episodes(show.id)[0]
    db.mark_sent(stored.id, "telegram")

    db.remove_show(show.id)
    assert db.get_show(show.id) is None
    assert db.list_episodes(show.id) == []


def test_episode_code_formats_season_and_number(db):
    show = add_show(db, 1, "Первый")
    db.upsert_episodes(show.id, [episode(2026, 193, NOW), episode(1, 7, NOW)])
    codes = {item.code for item in db.list_episodes(show.id)}
    assert codes == {"S2026E193", "S01E07"}


def test_real_fixture_episodes_load_into_database(db):
    """Проверка на настоящем ответе TVmaze, а не на собранном вручную."""
    from series_watcher.models import EpisodeRecord
    from series_watcher.schedule import TvMazeSchedule
    from series_watcher.tvmaze import dedupe_episodes, parse_episode

    payload = load_fixture("episodes_running")
    show = add_show(db, 75194, "Beyond the Gates")
    episodes = [
        EpisodeRecord(
            season=ep.season,
            number=ep.number,
            name=ep.name,
            aired_at=ep.airstamp,
            runtime=ep.runtime,
            summary=ep.summary,
            external_id=str(ep.id) if ep.id else None,
        )
        for ep in dedupe_episodes(parse_episode(item, 75194) for item in payload)
    ]
    added, _ = db.upsert_episodes(show.id, episodes)

    assert added == len(payload)
    nxt = db.next_episode(show.id, after=NOW)
    assert nxt is not None
    assert nxt.season == 2026
    assert nxt.number == 193
    assert nxt.aired_at.tzinfo is timezone.utc
    assert "<p>" not in (nxt.summary or "")