"""Рекомендации: авторство, жанры, отсев и отказы."""

from __future__ import annotations

import pytest

from series_watcher.config import Config
from series_watcher.db import ALREADY_WATCHED, NOT_INTERESTED, Database
from series_watcher.recommend import Recommender
from series_watcher.tmdb import TmdbClient, TmdbError
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture


def show_payload(tvmaze_id: int, name: str, **fields) -> dict:
    payload = dict(load_fixture("show"), id=tvmaze_id, name=name)
    payload.update(fields)
    return payload


@pytest.fixture
def parts(tmp_path, fake_tvmaze, sleeps):
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), cache=db, sleep=sleeps.append, rate_limit=(0, 10.0)
    )
    config = Config(notify_quiet_hours=None)
    yield db, client, config, fake_tvmaze
    client.close()
    db.close()


SCHEDULE_DAY = "2026-10-19"


@pytest.fixture
def recommender(parts):
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", load_fixture("schedule"))
    build = lambda **kw: Recommender(  # noqa: E731 - короткая локальная сборка
        config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,), **kw
    )
    return build, db, fake


def pin(fake, day: str, payload) -> None:
    fake.add(f"/schedule?date={day}", payload)


def track(recommender, tvmaze_id: int = 44933):
    instance, db, fake = recommender
    db.upsert_show(_parse(fake, tvmaze_id))
    return instance


def _parse(fake, tvmaze_id: int):
    from series_watcher.tvmaze import parse_show

    payload = load_fixture("show") if tvmaze_id == 44933 else dict(load_fixture("show"), id=tvmaze_id)
    return parse_show(payload)


def test_no_tracked_shows_means_no_recommendations(recommender):
    build, _, _ = recommender
    assert build().candidates() == []


def test_genres_alone_give_candidates(recommender):
    build, db, _ = recommender
    track(recommender)

    items = build().candidates(use_tmdb=False)

    assert items
    assert all(item.via in {"genre", "creator", "tmdb"} for item in items)
    assert all(item.reasons for item in items)
    assert all("жанры" in "; ".join(item.reasons) or "создатель" in "; ".join(item.reasons) for item in items)


def test_creator_of_tracked_show_is_not_offered_again(recommender):
    build, db, fake = recommender
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    items = build().candidates(use_tmdb=False)

    assert 44933 not in {item.show.id for item in items}


def test_other_creations_are_offered(recommender):
    build, db, fake = recommender
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    items = build().candidates(use_tmdb=False)

    assert [item.show.id for item in items] == [777]
    assert items[0].via == "creator"
    assert "создатель: Dan Erickson" in items[0].reasons


def test_rejects_are_excluded_forever(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    pin(fake, SCHEDULE_DAY, [])

    assert [item.show.id for item in build().candidates(use_tmdb=False)] == [777]

    db.reject(777, NOT_INTERESTED, title="Реванш")

    assert build().candidates(use_tmdb=False) == []


def test_rating_threshold_filters_candidates(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 5.0}))
    pin(fake, SCHEDULE_DAY, [])

    assert build().candidates(use_tmdb=False) == []
    assert [item.show.id for item in build().candidates(use_tmdb=False, min_rating=4.0)] == [777]


def test_only_scripted_shows_are_recommended(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", type="Miniseries", genres=["Drama"], rating={"average": 9.0}))
    pin(fake, SCHEDULE_DAY, [])

    assert build().candidates(use_tmdb=False) == []


def test_tmdb_director_moves_candidate_up(parts):
    """Режиссёр из TMDB — второй уровень: кандидат поднимается, а не выдумывается."""
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    db.upsert_show(_parse(fake, 44933))

    candidate = show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0})
    pin(fake, SCHEDULE_DAY, [{**load_fixture("schedule")[0], "show": candidate}])
    fake.add("/shows/777/crew", [
        {"type": "Director", "person": {"id": 1, "name": "Dan Erickson"}}
    ])

    plain = Recommender(config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,))
    without = plain.candidates(use_tmdb=False)[0]

    class FakeTmdb:
        def available(self):
            return True

        def find_by_tvdb(self, tvdb_id):
            return 12345

        def credits(self, tmdb_id):
            from series_watcher.tmdb import Credits, Person

            return Credits(directors=(Person(1, "Dan Erickson", "Director"),))

    with_tmdb = Recommender(
        config=config, db=db, tvmaze=client, tmdb=FakeTmdb(), schedule_dates=(SCHEDULE_DAY,)
    )
    item = with_tmdb.candidates()[0]
    assert without.via == "genre"
    assert without.reasons == ("жанры: Drama",)
    assert item.via == "tmdb"
    assert item.score > without.score
    assert "режиссёры и сценаристы: dan erickson" in item.reasons


def test_tmdb_without_key_changes_nothing(parts):
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", load_fixture("schedule"))
    db.upsert_show(_parse(fake, 44933))

    recommender = Recommender(
        config=config, db=db, tvmaze=client, tmdb=TmdbClient(None), schedule_dates=(SCHEDULE_DAY,)
    )

    items = recommender.candidates()
    assert items
    assert all("режиссёры" not in "; ".join(item.reasons) for item in items)


def test_recommendation_line_has_facts_from_api(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama", "Mystery"], rating={"average": 8.5}))
    pin(fake, SCHEDULE_DAY, [])

    line = build().candidates(use_tmdb=False)[0].line()

    assert line == "Реванш (Drama, Mystery) рейтинг 8.5"


def test_tracked_shows_are_kept_in_the_pool_for_context(recommender):
    """Пул собирается один раз; отслеживаемые туда попадают, чтобы не терять контекст."""
    build, db, fake = recommender
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    pool = build()._pool(db.list_shows(), {}, pool_size=5, schedule_days=3)

    assert [show.id for show in pool] == [44933]

class FakeTmdbApi:
    """Подмена api.themoviedb.org: отвечает на два нужных пути."""

    def __init__(self, find=None, credits=None, *, status: int = 200) -> None:
        self.find = find if find is not None else {"tv_results": [{"id": 12345, "name": "Severance"}]}
        self.credits = credits if credits is not None else {"crew": []}
        self.status = status
        self.calls: list[str] = []

    def transport(self):
        import httpx

        payload = {"find": self.find, "credits": self.credits}
        status = self.status
        calls = self.calls

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            key = "find" if request.url.path.startswith("/3/find/") else "credits"
            return httpx.Response(status, json=payload[key])

        return httpx.MockTransport(handler)


def test_tmdb_links_show_by_tvdb_id():
    api = FakeTmdbApi()
    client = TmdbClient("key", transport=api.transport())

    assert client.find_by_tvdb(371980) == 12345
    assert client.find_by_tvdb(None) is None
    assert api.calls == ["/3/find/371980"]


def test_tmdb_credits_split_by_job():
    api = FakeTmdbApi(credits={"crew": [
        {"id": 1, "name": "Dan Erickson", "job": "Director", "department": "Directing"},
        {"id": 2, "name": "Peter Herman", "job": "Writer", "department": "Writing"},
        {"id": 3, "name": "Adam MacKay", "job": "Director Of Photography", "department": "Camera"},
        {"id": 4, "name": "Someone", "job": "Actor", "department": "Acting"},
    ]})
    client = TmdbClient("key", transport=api.transport())

    credits = client.credits(12345)

    assert [person.name for person in credits.directors] == ["Dan Erickson", "Adam MacKay"]
    assert [person.name for person in credits.writers] == ["Peter Herman"]
    assert credits.names() == {"dan erickson", "adam mackay", "peter herman"}


def test_tmdb_without_key_makes_no_request():
    api = FakeTmdbApi()
    client = TmdbClient(None, transport=api.transport())

    assert client.available() is False
    with pytest.raises(TmdbError, match="TMDB_API_KEY"):
        client.credits(1)
    assert api.calls == []


def test_tmdb_bad_key_is_reported_plainly():
    api = FakeTmdbApi(status=401)
    client = TmdbClient("wrong", transport=api.transport())

    with pytest.raises(TmdbError, match="TMDB_API_KEY"):
        client.find_by_tvdb(371980)


def test_tmdb_credits_are_cached(tmp_path):
    api = FakeTmdbApi(credits={"crew": [{"id": 1, "name": "Dan Erickson", "job": "Director"}]})
    db = Database(tmp_path / "test.db")
    client = TmdbClient("key", cache=db, transport=api.transport())

    client.credits(12345)
    client.credits(12345)

    assert api.calls == ["/3/tv/12345/credits"]
    db.close()
