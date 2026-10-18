"""Рекомендации: авторство, жанры, отсев и отказы."""

from __future__ import annotations

import pytest

from series_watcher.config import Config
from series_watcher.db import ALREADY_WATCHED, NOT_INTERESTED, Database
from series_watcher.recommend import Recommender
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

    items = build().candidates()

    assert items
    assert all(item.via in {"жанры", "авторство"} for item in items)
    assert all(item.reasons for item in items)
    assert all("жанры" in "; ".join(item.reasons) or "создатель" in "; ".join(item.reasons) for item in items)


def test_creator_of_tracked_show_is_not_offered_again(recommender):
    build, db, fake = recommender
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    items = build().candidates()

    assert 44933 not in {item.show.id for item in items}


def test_other_creations_are_offered(recommender):
    build, db, fake = recommender
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    items = build().candidates()

    assert [item.show.id for item in items] == [777]
    assert items[0].via == "авторство"
    assert "создатель: Dan Erickson" in items[0].reasons


def test_rejects_are_excluded_forever(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    pin(fake, SCHEDULE_DAY, [])

    assert [item.show.id for item in build().candidates()] == [777]

    db.reject(777, NOT_INTERESTED, title="Реванш")

    assert build().candidates() == []


def test_rating_threshold_filters_candidates(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 5.0}))
    pin(fake, SCHEDULE_DAY, [])

    assert build().candidates() == []
    assert [item.show.id for item in build().candidates(min_rating=4.0)] == [777]


def test_only_scripted_shows_are_recommended(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", type="Miniseries", genres=["Drama"], rating={"average": 9.0}))
    pin(fake, SCHEDULE_DAY, [])

    assert build().candidates() == []






def test_recommendation_line_has_facts_from_api(recommender):
    build, db, fake = recommender
    track(recommender)
    fake.add("/people/169453/crewcredits", [
        {"type": "Creator", "_links": {"show": {"href": "https://api.tvmaze.com/shows/777", "name": "Реванш"}}}
    ])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama", "Mystery"], rating={"average": 8.5}))
    pin(fake, SCHEDULE_DAY, [])

    line = build().candidates()[0].line()

    assert line == "Реванш (Drama, Mystery) рейтинг 8.5"


def test_tracked_shows_are_kept_in_the_pool_for_context(recommender):
    """Пул собирается один раз; отслеживаемые туда попадают, чтобы не терять контекст."""
    build, db, fake = recommender
    track(recommender)
    pin(fake, SCHEDULE_DAY, [])

    pool = build()._pool(db.list_shows(), {}, pool_size=5, schedule_days=3)

    assert [show.id for show in pool] == [44933]


def person_credits(*items: tuple[str, int, str]) -> list[dict]:
    """Настоящий формат `/people/{id}/crewcredits`: роль и ссылка на сериал."""
    return [
        {"type": role, "_links": {"show": {"href": f"https://api.tvmaze.com/shows/{sid}", "name": name}}}
        for role, sid, name in items
    ]


def test_director_and_writer_are_used(parts):
    """Второй уровень строится на TVmaze: роли есть в /shows/{id}/crew."""
    db, client, config, fake = parts
    fake.add(
        "/shows/44933/crew",
        [
            {"type": "Director", "person": {"id": 500, "name": "Adam McKay"}},
            {"type": "Writer", "person": {"id": 501, "name": "Peter Herman"}},
            {"type": "Actor", "person": {"id": 502, "name": "Someone Famous"}},
        ],
    )
    fake.add("/people/500/crewcredits", person_credits(("Director", 777, "Реванш")))
    fake.add("/people/501/crewcredits", person_credits(("Writer", 777, "Реванш"), ("Writer", 778, "Другое")))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", [])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    fake.add("/shows/778", show_payload(778, "Другое", genres=["Drama"], rating={"average": 7.0}))
    db.upsert_show(_parse(fake, 44933))

    items = Recommender(
        config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,)
    ).candidates()

    assert [item.show.id for item in items] == [777, 778]
    assert items[0].via == "авторство"
    assert items[0].reasons == ("режиссёр: Adam McKay", "сценарист: Peter Herman", "жанры: Drama")
    assert items[0].score > items[1].score


def test_actors_are_ignored(parts):
    """Актёры пересекаются у всего подряд — в сигналы они не попадают."""
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Actor", "person": {"id": 900, "name": "Star"}}])
    fake.add("/people/900/crewcredits", person_credits(("Actor", 777, "Реванш")))
    fake.add(
        f"/schedule?date={SCHEDULE_DAY}",
        [{**load_fixture("schedule")[0], "show": show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0})}],
    )
    db.upsert_show(_parse(fake, 44933))

    items = Recommender(
        config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,)
    ).candidates()

    assert [item.via for item in items] == ["жанры"]


def test_creator_outranks_director(parts):
    db, client, config, fake = parts
    fake.add(
        "/shows/44933/crew",
        [
            {"type": "Creator", "person": {"id": 501, "name": "Peter Herman"}},
            {"type": "Director", "person": {"id": 500, "name": "Adam McKay"}},
        ],
    )
    fake.add("/people/501/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/people/500/crewcredits", person_credits(("Director", 777, "Реванш")))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", [])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    db.upsert_show(_parse(fake, 44933))

    item = Recommender(
        config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,)
    ).candidates()[0]

    assert item.reasons[0] == "создатель: Peter Herman"


def test_crew_lookups_are_cached(parts):
    """Каждый человек спросен один раз: повторный подбор не ходит в сеть."""
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Director", "person": {"id": 500, "name": "Adam McKay"}}])
    fake.add("/people/500/crewcredits", person_credits(("Director", 777, "Реванш")))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", [])
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    db.upsert_show(_parse(fake, 44933))
    recommender = Recommender(config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,))

    recommender.candidates()
    calls_after_first = len(fake.calls)
    recommender.candidates()

    assert len(fake.calls) == calls_after_first


def test_roles_are_matched_by_family():
    """В TVmaze роли гранулярные: «Staff Writer» — это сценарист."""
    from series_watcher.recommend import role_kind

    assert role_kind("Creator") == "creator"
    assert role_kind("Executive Producer") == "producer"
    assert role_kind("Co-Producer") == "producer"
    assert role_kind("Supervising Producer") == "producer"
    assert role_kind("Staff Writer") == "writer"
    assert role_kind("Director") == "director"


def test_other_jobs_are_not_a_signal():
    from series_watcher.recommend import role_kind

    for role in ("Actor", "Editor", "Sound Designer", "Music", "Casting"):
        assert role_kind(role) is None


def test_assistant_director_and_dop_are_not_directors():
    """Другие должности с похожим названием — не режиссёры."""
    from series_watcher.recommend import role_kind

    assert role_kind("First Assistant Director") is None
    assert role_kind("Second Assistant Director") is None
    assert role_kind("Director Of Photography") is None


def test_real_crew_fixture_yields_people(parts):
    """Настоящий состав Severance: роли есть, и они распознаются."""
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", [])
    db.upsert_show(_parse(fake, 44933))
    recommender = Recommender(config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,))

    people = recommender._people(db.list_shows())

    kinds = {kind for person in people.values() for kind in person.roles}
    assert "creator" in kinds
    assert "producer" in kinds
    producers = [person for person in people.values() if person.roles == {"producer"}]
    assert len(producers) <= 5  # продюсеров у сериала десятки, берём первых пятерых


def test_focus_narrows_the_signals(parts):
    """«Похожие на X» — сигналы только по X, а не по всей библиотеке."""
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Director", "person": {"id": 1, "name": "Из Severance"}}])
    fake.add("/people/1/crewcredits", person_credits(("Director", 777, "Из-под Severance")))
    fake.add("/shows/75194/crew", [{"type": "Director", "person": {"id": 2, "name": "Из Gateway"}}])
    fake.add("/people/2/crewcredits", person_credits(("Director", 778, "Из Gateway")))
    fake.add(f"/schedule?date={SCHEDULE_DAY}", [])
    fake.add("/shows/777", show_payload(777, "Из-под Severance", genres=["Drama"], rating={"average": 8.0}))
    fake.add("/shows/778", show_payload(778, "Из Gateway", genres=["Drama"], rating={"average": 8.0}))
    db.upsert_show(_parse(fake, 44933))
    db.upsert_show(_parse(fake, 75194))
    recommender = Recommender(config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,))
    focus = db.get_show_by_tvmaze_id(44933)

    all_items = {item.show.id for item in recommender.candidates()}
    focused = {item.show.id for item in recommender.candidates(focus=focus)}

    assert all_items == {777, 778}
    assert focused == {777}  # «из Gateway» в разы не входит


def test_focus_never_recommends_something_tracked(parts):
    db, client, config, fake = parts
    db.upsert_show(_parse(fake, 44933))
    db.upsert_show(_parse(fake, 75194))
    recommender = Recommender(config=config, db=db, tvmaze=client, schedule_dates=(SCHEDULE_DAY,))

    items = recommender.candidates(focus=db.get_show_by_tvmaze_id(75194))

    assert 75194 not in {item.show.id for item in items}
