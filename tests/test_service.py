"""Сервис: поиск сериала, выбор из кандидатов, повторное добавление."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.config import Config
from series_watcher.db import Database
from series_watcher.service import SeriesWatcher
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def watcher(tmp_path, fake_tvmaze, sleeps) -> SeriesWatcher:
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(),
        cache=db,
        sleep=sleeps.append,
        rate_limit=(0, 10.0),
    )
    yield SeriesWatcher(config=Config(), db=db, tvmaze=client)
    client.close()
    db.close()


def show_payload(tvmaze_id: int, name: str, source: str = "show") -> dict:
    return dict(load_fixture(source), id=tvmaze_id, name=name)


def add_ambiguous_candidates(fake_tvmaze) -> None:
    """Запрос, под который TVmaze отдаёт несколько сериалов."""
    fake_tvmaze.add(
        "/search/shows?q=dark knight",
        [
            {"score": 0.81, "show": show_payload(11, "Dark")},
            {"score": 0.64, "show": show_payload(12, "Dark Matter")},
            {"score": 0.42, "show": show_payload(13, "Dark Angel")},
        ],
    )
    for tvmaze_id in (11, 12, 13):
        fake_tvmaze.add(f"/shows/{tvmaze_id}", show_payload(tvmaze_id, f"сериал {tvmaze_id}"))
        fake_tvmaze.add(f"/shows/{tvmaze_id}/episodes", load_fixture("episodes"))


def test_exact_name_match_is_added_without_asking(watcher):
    result = watcher.add_show("Severance")
    assert result.status == "added"
    assert result.show.name == "Severance"
    assert result.episodes == len(load_fixture("episodes"))
    assert watcher.db.list_shows()[0].thetvdb_id == 371980


def test_article_and_case_in_query_still_match(watcher, fake_tvmaze):
    fake_tvmaze.add("/search/shows?q=the severance", load_fixture("search_shows"))
    result = watcher.add_show("the severance")
    assert result.status == "added"
    assert result.show.name == "Severance"


def test_ambiguous_query_returns_candidates(watcher, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)
    result = watcher.add_show("dark knight")
    assert result.status == "needs_choice"
    assert [item.show.name for item in result.candidates] == ["Dark", "Dark Matter", "Dark Angel"]
    assert watcher.db.list_shows() == []


def test_pick_selects_candidate(watcher, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)

    chosen = watcher.add_show("dark knight", pick=2)
    assert chosen.status == "added"
    assert chosen.show.name == "Dark Matter"
    assert [item.name for item in watcher.db.list_shows()] == ["Dark Matter"]


def test_pick_out_of_range_asks_again(watcher, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)
    result = watcher.add_show("dark knight", pick=9)
    assert result.status == "needs_choice"
    assert watcher.db.list_shows() == []


def test_single_candidate_is_added_without_question(watcher, fake_tvmaze):
    fake_tvmaze.add(
        "/search/shows?q=stays",
        [{"score": 0.5, "show": show_payload(21, "Stays with Us")}],
    )
    fake_tvmaze.add("/shows/21", show_payload(21, "Stays with Us"))
    fake_tvmaze.add("/shows/21/episodes", load_fixture("episodes"))
    assert watcher.add_show("stays").status == "added"


def test_unknown_query_is_not_found(watcher, fake_tvmaze):
    fake_tvmaze.add("/search/shows?q=zzzqqq", [])
    result = watcher.add_show("zzzqqq")
    assert result.status == "not_found"
    assert watcher.db.list_shows() == []


def test_adding_twice_keeps_one_row_and_refreshes_episodes(watcher):
    first = watcher.add_show("Severance")
    second = watcher.add_show("Severance")

    assert first.status == "added"
    assert second.status == "exists"
    assert len(watcher.db.list_shows()) == 1
    assert "новых 0" in second.message


def test_add_by_id_does_not_search(watcher, fake_tvmaze):
    result = watcher.add_show_by_id(44933)
    assert result.status == "added"
    assert not [call for call in fake_tvmaze.calls if call.startswith("/search")]


def test_remove_and_refresh(watcher):
    watcher.add_show("Severance")
    show = watcher.db.list_shows()[0]

    refreshed = watcher.refresh_show(show.tvmaze_id)
    assert refreshed.status == "exists"
    assert watcher.refresh_show(999_999).status == "not_found"
    assert watcher.remove_show(show.tvmaze_id) is True
    assert watcher.remove_show(show.tvmaze_id) is False
    assert watcher.db.list_shows() == []


def test_list_shows_orders_by_next_episode(watcher):
    watcher.add_show("Severance")
    watcher.add_show_by_id(75194)

    items = watcher.list_shows(now=NOW)
    assert [item.show.name for item in items] == ["Beyond the Gates", "Severance"]
    assert items[0].next_episode is not None
    assert items[0].next_episode.airdate == "2026-10-19"
    assert items[1].next_episode is None


def test_candidate_label_carries_kind_status_and_network(watcher, fake_tvmaze):
    fake_tvmaze.add(
        "/search/shows?q=gates",
        [
            {"score": 0.7, "show": show_payload(75194, "Beyond the Gates", "show_running")},
            {"score": 0.3, "show": show_payload(75195, "Gates of Hell")},
        ],
    )
    result = watcher.add_show("gates")
    assert result.status == "needs_choice"
    assert result.candidates[0].label == "Beyond the Gates (Scripted, Running, CBS)"
    assert result.candidates[1].label == "Gates of Hell (Scripted, Running, Apple TV)"