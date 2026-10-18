"""Источник расписания: общий интерфейс и реализация на TVmaze."""

from __future__ import annotations

import pytest

from series_watcher.schedule import (
    ROLE_LABELS,
    ScheduleError,
    ShowNotFound,
    TvMazeSchedule,
    role_family,
)
from series_watcher.tvmaze import TvMazeClient, TvMazeError

from conftest import load_fixture


@pytest.fixture
def source(fake_tvmaze) -> TvMazeSchedule:
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), rate_limit=(0, 10.0), timeout=1
    )
    return TvMazeSchedule(client)


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("Creator", "creator"),
        ("Executive Producer", "producer"),
        ("Co-Producer", "producer"),
        ("Visual Effects Producer", "producer"),
        ("Supervising Producer", "producer"),
        ("Staff Writer", "writer"),
        ("Director", "director"),
        ("First Assistant Director", None),
        ("Second Assistant Director", None),
        ("Director Of Photography", None),
        ("Actor", None),
        ("Editor", None),
        ("Sound Designer", None),
    ],
)
def test_role_families(role, expected):
    assert role_family(role) == expected


def test_every_family_has_a_label_and_a_weight():
    from series_watcher.schedule import ROLE_WEIGHTS

    for family in ("creator", "director", "writer", "producer"):
        assert family in ROLE_LABELS
        assert family in ROLE_WEIGHTS


def test_search_returns_our_model_not_tvmaze_dicts(source):
    found = source.search("Severance")

    assert found
    record = found[0]
    assert record.external_id == "44933"
    assert record.title == "Severance"
    assert record.identifiers == {"tvdb": "371980", "imdb": "tt11280740"}
    assert "<p>" not in (record.summary or "")


def test_get_episodes_keeps_utc(source):
    episodes = source.get_episodes("75194")
    aired = [item.aired_at for item in episodes if item.aired_at]

    assert aired
    assert all(moment.tzinfo is not None for moment in aired)


def test_get_credits_keeps_families_only(source, fake_tvmaze):
    fake_tvmaze.add("/shows/44933/crew", load_fixture("crew_show"))
    credits = source.get_credits("44933")

    roles = {item.role for item in credits}
    assert roles <= {"creator", "director", "writer", "producer"}
    assert "creator" in roles
    assert all(item.person_name for item in credits)


def test_producers_are_capped(source, fake_tvmaze):
    """У сериала десятки продюсеров, толку от шестого уже нет."""
    fake_tvmaze.add(
        "/shows/44933/crew",
        [{"type": f"Producer {i}", "person": {"id": i, "name": f"P{i}"}} for i in range(12)],
    )
    credits = source.get_credits("44933")

    assert len([item for item in credits if item.role == "producer"]) == 5


def test_works_carry_the_other_show(source, fake_tvmaze):
    fake_tvmaze.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    works = source.get_works("169453")

    assert works
    assert all(item.show_external_id.isdigit() for item in works)
    assert {item.role for item in works} <= {"creator", "director", "writer", "producer"}


def test_source_errors_are_translated(source):
    with pytest.raises(ShowNotFound):
        source.get_show("1")


def test_bad_identifier_is_a_clear_error(source):
    with pytest.raises(ScheduleError, match="числом"):
        source.get_episodes("не число")


def test_credits_survive_a_failing_lookup(source, fake_tvmaze):
    """Состав — необязательная часть: нет эндпоинта, значит пусто, а не краш."""
    fake_tvmaze.add("/shows/44933/crew", {"error": "nope"}, status=404)
    assert source.get_credits("44933") == ()


def test_a_second_source_only_needs_the_same_interface():
    """Контракт источника — это шесть методов; больше реализация не обязана."""
    from series_watcher.schedule import ScheduleSource

    required = {"name", "search", "get_show", "get_episodes", "get_credits", "get_works"}
    declared = set(dir(ScheduleSource)) | set(getattr(ScheduleSource, "__annotations__", {}))
    assert required <= declared
    assert TvMazeSchedule.__mro__[1] is object  # реализация не наследует, а подставляется
