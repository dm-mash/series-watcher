"""Рекомендации: стратегия по авторству, отсев, отказы, фокус, реестр."""

from __future__ import annotations

import pytest

from series_watcher.config import Config
from series_watcher.db import ALREADY_WATCHED, NOT_INTERESTED, Database
from series_watcher.schedule import TvMazeSchedule
from series_watcher.strategies import (
    Recommendation,
    Request,
    get as get_strategy,
    names,
    register,
)
from series_watcher.strategies.crew import COLD_START_MIN_RATING, CrewStrategy
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture, pin_schedule

SCHEDULE_DAY = "2026-10-19"


def show_payload(tvmaze_id: int, name: str, **fields) -> dict:
    payload = dict(load_fixture("show"), id=tvmaze_id, name=name)
    payload.update(fields)
    return payload


def person_credits(*items: tuple[str, int, str]) -> list[dict]:
    """Настоящий формат `/people/{id}/crewcredits`: роль и ссылка на сериал."""
    return [
        {"type": role, "_links": {"show": {"href": f"https://api.tvmaze.com/shows/{sid}", "name": title}}}
        for role, sid, title in items
    ]


def _parse(fake, tvmaze_id: int = 44933):
    from series_watcher.tvmaze import parse_show

    payload = dict(load_fixture("show"), id=tvmaze_id) if tvmaze_id != 44933 else load_fixture("show")
    return parse_show(payload)


@pytest.fixture
def parts(tmp_path, fake_tvmaze, sleeps):
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), cache=db, sleep=sleeps.append, rate_limit=(0, 10.0)
    )
    config = Config(notify_quiet_hours=None)
    fake_tvmaze.add("/shows/44933/crew", load_fixture("crew_show"))
    fake_tvmaze.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake_tvmaze)
    yield db, client, config, fake_tvmaze
    client.close()
    db.close()


def run(parts, *, limit: int = 5, **kwargs):
    db, client, config, fake = parts
    request = Request(config=config, db=db, source=TvMazeSchedule(client), limit=limit, **kwargs)
    return list(get_strategy("crew").candidates(request))


def track(parts, tvmaze_id: int = 44933):
    db, client, config, fake = parts
    record = TvMazeSchedule(client)._show(_parse(fake, tvmaze_id))
    db.upsert_show("tvmaze", record)
    return record


def pin(fake, payload) -> None:
    pin_schedule(fake, payload)


# --- базовые случаи ---------------------------------------------------------


def test_no_tracked_shows_means_no_recommendations(parts):
    assert run(parts) == []


def test_genres_alone_give_candidates(parts):
    track(parts)
    items = run(parts)

    assert items
    assert all(item.via in {"жанры", "авторство"} for item in items)
    assert all(item.reasons for item in items)


def test_creator_of_tracked_show_is_not_offered_again(parts):
    track(parts)
    pin(parts[3], [])

    assert 44933 not in {item.show.external_id for item in run(parts)}


def test_other_creations_are_offered(parts):
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    track(parts)
    pin(fake, [])

    items = run(parts)

    assert [item.show.external_id for item in items] == ["777"]
    assert items[0].via == "авторство"
    assert "создатель: Dan Erickson" in items[0].reasons


def test_rejects_are_excluded_forever(parts):
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    track(parts)
    pin(fake, [])

    assert [item.show.external_id for item in run(parts)] == ["777"]

    db.reject("777", NOT_INTERESTED, title="Реванш")

    assert run(parts) == []


def test_rating_threshold_filters_candidates(parts):
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 5.0}))
    track(parts)
    pin(fake, [])

    assert run(parts) == []
    assert [item.show.external_id for item in run(parts, min_rating=4.0)] == ["777"]


def test_only_scripted_shows_are_recommended(parts):
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add(
        "/shows/777",
        show_payload(777, "Реванш", type="Miniseries", genres=["Drama"], rating={"average": 9.0}),
    )
    track(parts)
    pin(fake, [])

    assert run(parts) == []


def test_recommendation_line_has_facts_from_api(parts):
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama", "Mystery"], rating={"average": 8.5}))
    track(parts)
    pin(fake, [])

    assert run(parts)[0].line() == "Реванш (Drama, Mystery) рейтинг 8.5"


def test_tracked_shows_stay_in_the_pool_for_context(parts):
    track(parts)
    pin(parts[3], [])

    pool = get_strategy("crew")._pool(
        Request(config=parts[2], db=parts[0], source=TvMazeSchedule(parts[1])), parts[0].list_shows(), {}
    )

    assert [record.external_id for record in pool] == ["44933"]


# --- роли -------------------------------------------------------------------


def test_director_and_writer_are_used(parts):
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
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    fake.add("/shows/778", show_payload(778, "Другое", genres=["Drama"], rating={"average": 7.0}))
    track(parts)
    pin(fake, [])

    items = run(parts)

    assert [item.show.external_id for item in items] == ["777", "778"]
    assert items[0].via == "авторство"
    assert items[0].reasons == ("режиссёр: Adam McKay", "сценарист: Peter Herman", "жанры: Drama")
    assert items[0].score > items[1].score


def test_actors_are_ignored(parts):
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Actor", "person": {"id": 900, "name": "Star"}}])
    fake.add("/people/900/crewcredits", person_credits(("Actor", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    track(parts)
    pin(fake, [{**load_fixture("schedule")[0], "show": show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0})}])

    items = run(parts)

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
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    track(parts)
    pin(fake, [])

    item = run(parts)[0]

    assert item.reasons[0] == "создатель: Peter Herman"


def test_crew_lookups_are_cached(parts):
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Director", "person": {"id": 500, "name": "Adam McKay"}}])
    fake.add("/people/500/crewcredits", person_credits(("Director", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.0}))
    track(parts)
    pin(fake, [])

    run(parts)
    calls_after_first = len(fake.calls)
    run(parts)

    assert len(fake.calls) == calls_after_first


# --- фокус ------------------------------------------------------------------


def test_focus_narrows_the_signals(parts):
    db, client, config, fake = parts
    fake.add("/shows/44933/crew", [{"type": "Director", "person": {"id": 1, "name": "Из Severance"}}])
    fake.add("/people/1/crewcredits", person_credits(("Director", 777, "Из-под Severance")))
    fake.add("/shows/75194/crew", [{"type": "Director", "person": {"id": 2, "name": "Из Gateway"}}])
    fake.add("/people/2/crewcredits", person_credits(("Director", 778, "Из Gateway")))
    fake.add("/shows/777", show_payload(777, "Из-под Severance", genres=["Drama"], rating={"average": 8.0}))
    fake.add("/shows/778", show_payload(778, "Из Gateway", genres=["Drama"], rating={"average": 8.0}))
    track(parts)
    track(parts, 75194)
    pin(fake, [])

    everything = {item.show.external_id for item in run(parts)}
    focus = db.get_by_identifier("tvmaze", "44933")
    focused = {item.show.external_id for item in run(parts, focus=focus)}

    assert everything == {"777", "778"}
    assert focused == {"777"}


def test_focus_never_recommends_something_tracked(parts):
    db, client, config, fake = parts
    track(parts)
    track(parts, 75194)

    items = run(parts, focus=db.get_by_identifier("tvmaze", "75194"))

    assert "75194" not in {item.show.external_id for item in items}


# --- холодный старт ---------------------------------------------------------


def test_cold_start_offers_what_is_on_air(parts):
    db, client, config, fake = parts
    items = get_strategy("crew").cold_start(
        Request(config=config, db=db, source=TvMazeSchedule(client))
    )

    assert items
    assert all(item.via == "расписание" for item in items)
    assert all(item.show.kind == "Scripted" for item in items)
    assert all(any("канал" in reason or "жанры" in reason for reason in item.reasons) for item in items)


def test_cold_start_respects_rejects(parts):
    db, client, config, fake = parts
    strategy = get_strategy("crew")
    request = Request(config=config, db=db, source=TvMazeSchedule(client))
    first = strategy.cold_start(request, limit=1)[0]
    db.reject(first.show.external_id, NOT_INTERESTED, title=first.show.title)

    assert first.show.external_id not in {
        item.show.external_id for item in strategy.cold_start(request, limit=20)
    }


def test_cold_start_threshold_is_lower_than_usual():
    assert COLD_START_MIN_RATING < 6.5


# --- реестр -----------------------------------------------------------------


def test_registry_knows_the_crew_strategy():
    assert "crew" in names()
    assert isinstance(get_strategy(), CrewStrategy)


def test_unknown_strategy_is_a_clear_error():
    with pytest.raises(ValueError, match="неизвестный рекомендатель"):
        get_strategy("магия")


def test_another_strategy_needs_no_changes_in_the_service(parts):
    """Подключение нового рекомендателя — это только регистрация."""
    calls: list[object] = []

    class Magic:
        name = "magic"
        description = "всегда предлагает одно и то же"

        def candidates(self, request: Request):
            calls.append(request)
            return [Recommendation(show=request.source.get_show("44933"), score=1.0, via="magic")]

    register("magic", Magic)
    try:
        assert "magic" in names()
        db, client, config, _ = parts
        track(parts)
        items = list(get_strategy("magic").candidates(
            Request(config=config, db=db, source=TvMazeSchedule(client))
        ))
        assert [item.via for item in items] == ["magic"]
        assert len(calls) == 1
    finally:
        from series_watcher.strategies.registry import FACTORIES

        FACTORIES.pop("magic", None)


def test_reject_kinds_are_still_available(parts):
    db, _, _, _ = parts
    db.reject("999", ALREADY_WATCHED, title="Что-то")
    assert "tvmaze:999" in db.rejected_ids()


def _rated(tvmaze_id: int, name: str, rating: float, genres: list[str], status: str = "Running") -> dict:
    return {
        "id": tvmaze_id,
        "name": name,
        "type": "Scripted",
        "status": status,
        "genres": list(genres),
        "rating": {"average": rating},
        "image": None,
        "summary": None,
        "network": None,
        "webChannel": None,
        "premiered": "2026-09-01",
        "ended": None,
        "runtime": 45,
        "language": "English",
        "externals": {},
        "url": f"https://www.tvmaze.com/shows/{tvmaze_id}",
    }


def _aged(tvmaze_id: int, name: str, rating: float, genres: list[str]) -> dict:
    return _rated(tvmaze_id, name, rating, genres, status="Ended")


def _airing(*shows: dict) -> list[dict]:
    """Запись расписания TVmaze: сериал и серия, которая с ним идёт."""
    return [
        {"show": show, "season": 1, "number": 1, "name": "pilot", "airstamp": "2026-10-19T20:00:00+00:00"}
        for show in shows
    ]


def test_thin_pool_is_topped_up_by_rating(parts, fake_tvmaze):
    """Похожих не нашлось — добираем хорошие и честно помечаем их как добор."""
    db, client, config, fake = parts
    track(parts)
    pin_schedule(
        fake,
        _airing(
            _rated(1, "Unrelated Gem", 9.1, ["Документальный"]),
            _rated(2, "Okay Show", 7.0, ["Документальный"]),
        ),
    )

    items = CrewStrategy().candidates(
        Request(config=config, db=db, source=TvMazeSchedule(client), limit=3)
    )

    added = [item for item in items if item.via == "добор"]
    # По рейтингу: девять и десять выше, чем семь, независимо от порядка в пуле.
    assert [item.show.title for item in added] == ["Unrelated Gem", "Okay Show"]
    assert all("добор" in item.reasons[0] for item in added)


def test_top_up_gives_place_to_the_related_ones(parts, fake_tvmaze):
    """Сначала похожие, добором — только остаток места."""
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 6.6}))
    track(parts)
    # В расписании только чужой сериал с рейтингом выше: место забирает похожий.
    pin_schedule(fake, _airing(_rated(1, "Unrelated Gem", 9.1, ["Документальный"])))

    items = CrewStrategy().candidates(
        Request(config=config, db=db, source=TvMazeSchedule(client), limit=1)
    )

    assert [item.show.external_id for item in items] == ["777"]
    assert items[0].via == "авторство"


def test_focus_gets_no_unrelated_top_up(parts, fake_tvmaze):
    """«Похожее на X» — про X, а не про хорошие сериалы вообще."""
    from series_watcher.tvmaze import parse_show

    db, client, config, fake = parts
    track(parts)
    db.upsert_show("tvmaze", TvMazeSchedule(client)._show(parse_show(show_payload(44934, "Focus Show"))))
    focus = next(show for show in db.list_shows() if show.name == "Focus Show")
    pin_schedule(fake, _airing(_rated(1, "Unrelated Gem", 9.1, ["Документальный"])))

    items = CrewStrategy().candidates(
        Request(config=config, db=db, source=TvMazeSchedule(client), limit=3, focus=focus)
    )

    assert all(item.via != "добор" for item in items)


def test_authored_work_is_fetched_despite_a_big_schedule(parts, fake_tvmaze):
    """Раньше лимит считался от размера пула: при большом расписании ветка
    «работы наших авторов» не подгружалась никогда и рекомендации были только
    жанровыми. Лимит теперь ограничивает число запросов, а не размер пула."""
    db, client, config, fake = parts
    fake.add("/people/169453/crewcredits", person_credits(("Creator", 777, "Реванш")))
    fake.add("/shows/777", show_payload(777, "Реванш", genres=["Drama"], rating={"average": 8.5}))
    track(parts)
    pin_schedule(
        fake,
        _airing(*[_rated(100 + n, f"Шум {n}", 9.9 - n / 10, ["Комедия"]) for n in range(20)]),
    )

    items = run(parts, limit=5)

    # Похожий по авторству — первым, добором ниже по рейтингу.
    assert [item.via for item in items] == ["авторство", "добор", "добор", "добор", "добор"]
    assert items[0].show.external_id == "777"


def test_ended_shows_are_not_recommended(parts, fake_tvmaze):
    """Трекер ждёт следующего эпизода: законченному сериалу тут не место.

    Раньше такие вылезали по общему продюсеру — The Pacific (2002) в подборке к
    Slow Horses выглядел как ошибка.
    """
    db, client, config, fake = parts
    track(parts)
    pin_schedule(
        fake,
        _airing(
            _aged(1, "Old War Drama", 9.4, ["Drama"]),
            _rated(2, "New Drama", 7.0, ["Drama"]),
        ),
    )

    items = CrewStrategy().candidates(
        Request(config=config, db=db, source=TvMazeSchedule(client), limit=5)
    )

    assert [item.show.title for item in items] == ["New Drama"]


def test_unknown_status_is_treated_as_alive(parts, fake_tvmaze):
    """Нет данных о статусе — это не «закончен»."""
    db, client, config, fake = parts
    track(parts)
    payload = _airing(_rated(1, "Без статуса", 7.5, ["Drama"]))
    payload[0]["show"]["status"] = None
    pin_schedule(fake, payload)

    items = CrewStrategy().candidates(
        Request(config=config, db=db, source=TvMazeSchedule(client), limit=5)
    )

    assert [item.show.title for item in items] == ["Без статуса"]


def test_cold_start_skips_ended_shows(parts, fake_tvmaze):
    db, client, config, fake = parts
    pin_schedule(fake, _airing(_aged(1, "Old War Drama", 9.4, ["Drama"])))

    items = CrewStrategy().cold_start(
        Request(config=config, db=db, source=TvMazeSchedule(client)), limit=3
    )

    assert list(items) == []
