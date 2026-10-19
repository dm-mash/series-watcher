"""Сервис: поиск сериала, выбор из кандидатов, повторное добавление."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.config import Config
from series_watcher.db import Database
from series_watcher.llm import LLMClient
from series_watcher.nlu import Intent, rules
from series_watcher.service import SeriesWatcher
from series_watcher.schedule import TvMazeSchedule
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture
from test_llm import FakeLLM, reply

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
    yield SeriesWatcher(config=Config(), db=db, source=TvMazeSchedule(client))
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
    assert "tvdb:371980" in {i.key for i in watcher.db.list_shows()[0].identifiers}


def test_article_and_case_in_query_still_match(watcher, fake_tvmaze):
    fake_tvmaze.add("/search/shows?q=the severance", load_fixture("search_shows"))
    result = watcher.add_show("the severance")
    assert result.status == "added"
    assert result.show.name == "Severance"


def test_ambiguous_query_returns_candidates(watcher, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)
    result = watcher.add_show("dark knight")
    assert result.status == "needs_choice"
    assert [item.show.title for item in result.candidates] == ["Dark", "Dark Matter", "Dark Angel"]
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

    refreshed = watcher.refresh_show(show.key.external_id)
    assert refreshed.status == "exists"
    assert watcher.refresh_show(999_999).status == "not_found"
    assert watcher.remove_show(show.key.external_id) is True
    assert watcher.remove_show(show.key.external_id) is False
    assert watcher.db.list_shows() == []


def test_list_shows_orders_by_next_episode(watcher):
    watcher.add_show("Severance")
    watcher.add_show_by_id(75194)

    items = watcher.list_shows(now=NOW)
    assert [item.show.name for item in items] == ["Beyond the Gates", "Severance"]
    assert items[0].next_episode is not None
    assert items[0].next_episode.aired_at == datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)
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

def llm_for(fake: FakeLLM) -> LLMClient:
    config = Config.from_env(
        env_file=None, environ={"LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "model-1"}
    )
    return LLMClient(config, transport=fake.transport())


def test_description_is_resolved_into_a_show(watcher, fake_tvmaze):
    """Пользователь описал сериал словами: поиск по подстроке не помогает."""
    fake_tvmaze.add("/search/shows?q=where people get their memories cut", [])
    watcher.llm = llm_for(
        FakeLLM(reply("Severance"), reply('{"id": 44933, "confidence": 0.95}'))
    )

    result = watcher.add_show("where people get their memories cut")

    assert result.status == "added"
    assert result.show.name == "Severance"
    assert "нашёл по описанию" in result.message


def test_ambiguous_candidates_are_resolved_by_the_model(watcher, fake_tvmaze):
    fake_tvmaze.add(
        "/search/shows?q=dark knight",
        [
            {"score": 0.8, "show": show_payload(11, "Dark")},
            {"score": 0.5, "show": show_payload(12, "Dark Matter")},
        ],
    )
    fake_tvmaze.add("/shows/12", show_payload(12, "Dark Matter"))
    fake_tvmaze.add("/shows/12/episodes", load_fixture("episodes"))
    watcher.llm = llm_for(FakeLLM(reply("2")))

    result = watcher.add_show("dark knight")

    assert result.status == "added"
    assert result.show.name == "Dark Matter"


def test_unsure_model_leaves_the_choice_to_the_user(watcher, fake_tvmaze):
    fake_tvmaze.add(
        "/search/shows?q=dark knight",
        [
            {"score": 0.8, "show": show_payload(11, "Dark")},
            {"score": 0.5, "show": show_payload(12, "Dark Matter")},
        ],
    )
    watcher.llm = llm_for(FakeLLM(reply("НЕ УВЕРЕН")))

    result = watcher.add_show("dark knight")

    assert result.status == "needs_choice"
    assert len(result.candidates) == 2


def test_episode_brief_is_written_once_and_remembered(watcher):
    watcher.add_show_by_id(44933)
    episode = next(item for item in watcher.db.list_episodes(watcher.db.list_shows()[0].id) if item.summary)
    fake = FakeLLM(reply("Байерс снова вскрывает дело."), reply("Второй вызов"))
    watcher.llm = llm_for(fake)

    first = watcher.episode_brief(episode)
    second = watcher.episode_brief(episode)

    assert first == "Байерс снова вскрывает дело."
    assert second == first
    assert len(fake.requests) == 1


def test_without_model_brief_is_just_absent(watcher):
    watcher.add_show_by_id(44933)
    episode = next(item for item in watcher.db.list_episodes(watcher.db.list_shows()[0].id) if item.summary)

    assert watcher.episode_brief(episode) is None


def test_service_replies_carry_no_attribution(watcher):
    """Подпись TVmaze — только там, где в ответе есть данные TVmaze."""
    watcher.add_show_by_id(44933)
    from series_watcher.nlu import UNKNOWN, WHAT_NEW, WHERE_TO_WATCH

    empty_news = watcher.handle(Intent(WHAT_NEW)).text
    unknown = watcher.handle(Intent(UNKNOWN)).text
    empty_where = watcher.handle(Intent(WHERE_TO_WATCH)).text

    assert "Данные: TVmaze" not in empty_news
    assert "Данные: TVmaze" not in unknown
    assert "Данные: TVmaze" not in empty_where


def test_data_replies_keep_attribution(watcher):
    watcher.add_show_by_id(75194)  # идёт в эфире, в ответе есть названия и время
    digest = watcher.digest(days=7, now=NOW)

    assert "Данные: TVmaze" in digest
    assert "Beyond the Gates" in digest


def test_about_asks_by_description_when_name_finds_nothing(watcher, fake_tvmaze):
    """«что за сериал, где людям режут память» — это описание, а не название.

    Раньше такой вопрос всегда заканчивался «не нашёл», хотя описание разобрать
    было можно: поиск по названию падал, и запасного прохода не было.
    """
    fake_tvmaze.add("/search/shows?q=где людям режут память", [])
    fake_tvmaze.add(
        "/search/shows?q=Westworld",
        [{"score": 0.9, "show": dict(load_fixture("show"), id=1, name="Westworld")}],
    )
    fake_tvmaze.add("/shows/1", dict(load_fixture("show"), id=1, name="Westworld"))
    watcher.llm = llm_for(FakeLLM(reply("Westworld")))

    answer = watcher.handle(rules("что за сериал где людям режут память"))

    assert "Westworld" in answer.text


def test_llm_trouble_is_not_reported_as_not_found(watcher, fake_tvmaze):
    """Сбой модели и отсутствие сериала — разные вещи, и говорить надо что есть."""
    from series_watcher.llm import LLMTruncated

    fake_tvmaze.add("/search/shows?q=где людям режут память", [])
    watcher.llm = llm_for(FakeLLM(400))

    answer = watcher.handle(rules("что за сериал где людям режут память"))

    assert "не нашёл" not in answer.text
