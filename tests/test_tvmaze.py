"""Разбор ответов TVmaze и поведение клиента: кэш, ретраи, лимит запросов."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.tvmaze import (
    RateLimiter,
    TvMazeClient,
    TvMazeError,
    TvMazeNotFound,
    dedupe_episodes,
    normalize_name,
    parse_airstamp,
    parse_episode,
    parse_show,
    strip_html,
)

from conftest import FakeTVmaze, load_fixture


def make_client(fake: FakeTVmaze, sleeps: list[float], **kwargs) -> TvMazeClient:
    options = dict(
        transport=fake.transport(),
        sleep=sleeps.append,
        rate_limit=(0, 10.0),
    )
    options.update(kwargs)
    return TvMazeClient(**options)


def test_parse_show_takes_network_from_web_channel():
    """У Severance поле `network` пустое, канал лежит в `webChannel`."""
    show = parse_show(load_fixture("show"))
    assert show.id == 44933
    assert show.name == "Severance"
    assert show.network == "Apple TV"
    assert show.kind == "Scripted"
    assert show.status == "Running"
    assert show.thetvdb_id == 371980
    assert show.imdb_id == "tt11280740"
    assert show.genres == ("Drama", "Science-Fiction", "Mystery")
    assert show.updated == load_fixture("show")["updated"]


def test_summary_is_stripped_from_html():
    show = parse_show(load_fixture("show"))
    assert show.summary is not None
    assert "<p>" not in show.summary
    assert show.summary.startswith("Mark Scout leads a team at Lumon Industries")


def test_episode_time_comes_from_airstamp_not_local_fields():
    """`airdate` и `airtime` местные: 14:00 по местному это 18:00Z."""
    raw = next(
        item
        for item in load_fixture("episodes_running")
        if item["airdate"] == "2026-10-19" and item["airtime"] == "14:00"
    )
    episode = parse_episode(raw, show_id=75194)
    assert episode.airstamp == datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)
    assert episode.airdate == "2026-10-19"
    assert episode.airtime == "14:00"


def test_airstamp_wins_when_local_date_and_time_look_like_another_day():
    episode = parse_episode(
        {
            "season": 1,
            "number": 5,
            "airdate": "2026-10-19",
            "airtime": "00:00",
            "airstamp": "2026-10-20T04:00:00+00:00",
        },
        show_id=1,
    )
    assert episode.airstamp == datetime(2026, 10, 20, 4, 0, tzinfo=timezone.utc)
    assert episode.airstamp.date() != datetime(2026, 10, 19).date()


def test_airstamp_without_value_stays_unknown():
    assert parse_airstamp(None) is None
    assert parse_airstamp("") is None
    assert parse_airstamp("вчера") is None


def test_client_counts_requests_it_makes(fake_tvmaze, sleeps):
    client = make_client(fake_tvmaze, sleeps)
    assert client.get_show(44933).name == "Severance"
    with pytest.raises(TvMazeError):
        client.get_show(1)
    assert client.request_count == 2


def test_dedupe_keeps_one_episode_per_season_and_number():
    payload = load_fixture("episodes")
    doubled = payload + payload[:3]
    episodes = dedupe_episodes(parse_episode(item, 44933) for item in doubled)
    keys = [(item.season, item.number) for item in episodes]
    assert len(keys) == len(set(keys))
    assert len(episodes) == len(payload)


def test_cache_in_database_prevents_second_request(tmp_path, fake_tvmaze, sleeps):
    from series_watcher.db import Database

    db = Database(tmp_path / "test.db")
    client = make_client(fake_tvmaze, sleeps, cache=db, ttl=21600.0)

    assert client.get_show(44933).id == 44933
    assert client.get_show(44933).id == 44933
    assert fake_tvmaze.calls_to("/shows/44933") == 1
    db.close()


def test_stale_cache_is_refetched(tmp_path, fake_tvmaze, sleeps):
    from series_watcher.db import Database

    db = Database(tmp_path / "test.db", clock=lambda: "2000-01-01T00:00:00+00:00")
    client = make_client(fake_tvmaze, sleeps, cache=db, ttl=60.0)

    client.get_show(44933)
    client.get_show(44933)
    assert fake_tvmaze.calls_to("/shows/44933") == 2
    db.close()


def test_refresh_skips_cache(tmp_path, fake_tvmaze, sleeps):
    from series_watcher.db import Database

    db = Database(tmp_path / "test.db")
    client = make_client(fake_tvmaze, sleeps, cache=db)
    client.get_show(44933)
    client.get_show(44933, refresh=True)
    assert fake_tvmaze.calls_to("/shows/44933") == 2
    db.close()


def test_episodes_are_deduplicated_by_season_and_number(fake_tvmaze, sleeps):
    client = make_client(fake_tvmaze, sleeps)
    episodes = client.get_episodes(44933)
    assert len(episodes) == len(load_fixture("episodes"))
    assert episodes[0].airstamp is not None


def test_search_returns_results_sorted_by_score(fake_tvmaze, sleeps):
    client = make_client(fake_tvmaze, sleeps)
    results = client.search_shows("Severance")
    assert results
    assert results[0].show.name == "Severance"
    assert [item.score for item in results] == sorted(
        (item.score for item in results), reverse=True
    )


def test_schedule_is_parsed_into_show_and_episode(fake_tvmaze, sleeps):
    client = make_client(fake_tvmaze, sleeps)
    entries = client.get_schedule("2026-10-19")
    assert entries
    assert any(entry.episode.airstamp is not None for entry in entries)
    assert all(entry.episode.show_id == entry.show.id for entry in entries)


def test_retries_on_server_error(fake_tvmaze, sleeps):
    fake_tvmaze.add("/shows/44933", {"error": "boom"}, status=503)
    client = make_client(fake_tvmaze, sleeps, retries=2)
    with pytest.raises(TvMazeError):
        client.get_show(44933)
    assert fake_tvmaze.calls_to("/shows/44933") == 3
    assert len(sleeps) == 2


def test_retry_succeeds_after_transient_failure(fake_tvmaze, sleeps):
    fake_tvmaze.add("/shows/44933", {"error": "boom"}, status=500)
    client = make_client(fake_tvmaze, sleeps, retries=3)
    with pytest.raises(TvMazeError):
        client.get_show(44933)
    fake_tvmaze.add("/shows/44933", load_fixture("show"))
    assert client.get_show(44933).name == "Severance"


def test_retry_after_header_is_respected(fake_tvmaze, sleeps):
    fake_tvmaze.add(
        "/shows/44933", {"error": "slow down"}, status=429, headers={"Retry-After": "3"}
    )
    client = make_client(fake_tvmaze, sleeps, retries=1)
    with pytest.raises(TvMazeError):
        client.get_show(44933)
    assert sleeps == [3.0]


def test_missing_show_raises_not_found(fake_tvmaze, sleeps):
    client = make_client(fake_tvmaze, sleeps)
    with pytest.raises(TvMazeNotFound):
        client.get_show(1)


def test_rate_limiter_keeps_tvmaze_window():
    now = [0.0]
    slept: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, 10.0, clock=clock, sleep=sleep)
    for _ in range(5):
        limiter.acquire()
    assert sum(slept) == pytest.approx(20.0)


def test_client_never_exceeds_limit_per_window(fake_tvmaze, sleeps):
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = TvMazeClient(
        transport=fake_tvmaze.transport(),
        sleep=sleep,
        rate_limit=(2, 10.0),
        clock=clock,
    )
    for _ in range(3):
        client.get_show(44933, refresh=True)
    assert now[0] >= 10.0


def test_normalize_name_drops_case_articles_and_spaces():
    assert normalize_name("  The   Office ") == "office"
    assert normalize_name("Severance") == "severance"


def test_strip_html_keeps_text_and_entities():
    assert strip_html("<p>Hello &amp; <b>bye</b></p>") == "Hello & bye"
    assert strip_html(None) is None