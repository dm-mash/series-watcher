"""CLI: добавление, выбор из кандидатов, список с переводом в зону пользователя."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest

from series_watcher.cli import EXIT_ERROR, EXIT_NEEDS_CHOICE, EXIT_OK, main
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture


@pytest.fixture
def cli(monkeypatch, tmp_path, fake_tvmaze, sleeps):
    monkeypatch.chdir(tmp_path)

    def factory(**kwargs) -> TvMazeClient:
        kwargs["transport"] = fake_tvmaze.transport()
        kwargs["sleep"] = sleeps.append
        kwargs["rate_limit"] = (0, 10.0)
        return TvMazeClient(**kwargs)

    monkeypatch.setattr("series_watcher.cli.TvMazeClient", factory)

    def run(*argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = main(["--db", str(tmp_path / "test.db"), *argv], stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    return run


def show_payload(tvmaze_id: int, name: str) -> dict:
    return dict(load_fixture("show"), id=tvmaze_id, name=name)


def add_ambiguous_candidates(fake_tvmaze) -> None:
    fake_tvmaze.add(
        "/search/shows?q=dark knight",
        [
            {"score": 0.8, "show": show_payload(11, "Dark")},
            {"score": 0.5, "show": show_payload(12, "Dark Matter")},
        ],
    )
    fake_tvmaze.add("/shows/12", show_payload(12, "Dark Matter"))
    fake_tvmaze.add("/shows/12/episodes", load_fixture("episodes"))


def test_add_prints_show_and_credit(cli):
    code, out, _ = cli("add", "Severance")
    assert code == EXIT_OK
    assert "Добавлен: Severance" in out
    assert "эпизодов в базе: 19" in out
    assert "https://www.tvmaze.com" in out


def test_add_without_query_or_id_is_an_error(cli):
    code, _, err = cli("add")
    assert code == EXIT_ERROR
    assert "название сериала или --id" in err


def test_unknown_show_reports_failure(cli, fake_tvmaze):
    fake_tvmaze.add("/search/shows?q=zzzqqq", [])
    code, _, err = cli("add", "zzzqqq")
    assert code == EXIT_ERROR
    assert "zzzqqq" in err


def test_ambiguous_query_asks_for_pick(cli, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)
    code, _, err = cli("add", "dark knight")
    assert code == EXIT_NEEDS_CHOICE
    assert "1. Dark (Scripted, Running, Apple TV)" in err
    assert "2. Dark Matter (Scripted, Running, Apple TV)" in err
    assert "--pick N" in err


def test_pick_finishes_the_add(cli, fake_tvmaze):
    add_ambiguous_candidates(fake_tvmaze)
    code, out, _ = cli("add", "dark knight", "--pick", "2")
    assert code == EXIT_OK
    assert "Добавлен: Dark Matter" in out


def test_empty_list_suggests_add(cli):
    code, out, _ = cli("list")
    assert code == EXIT_OK
    assert "Список пуст" in out
    assert "TELEGRAM_BOT_TOKEN" in out


def test_list_shows_next_episode_in_user_timezone(cli):
    cli("add", "--id", "75194")

    _, moscow, _ = cli("--tz", "Europe/Moscow", "list")
    _, new_york, _ = cli("--tz", "America/New_York", "list")
    _, utc, _ = cli("--tz", "UTC", "list")

    assert "19.10.2026 21:00" in moscow
    assert "19.10.2026 14:00" in new_york
    assert "19.10.2026 18:00" in utc
    assert "время: Europe/Moscow" in moscow


def test_list_keeps_shows_without_planned_episodes(cli):
    cli("add", "Severance")
    code, out, _ = cli("list")
    assert code == EXIT_OK
    assert "Severance" in out
    assert "следующий эпизод: не запланирован" in out
    assert "всего эпизодов: 19" in out


def test_list_says_what_is_disabled(cli):
    cli("add", "Severance")
    _, out, _ = cli("list")
    assert "Что отключено:" in out
    assert "нет TMDB_API_KEY" in out
    assert "нет JUSTWATCH_API_KEY" in out


def test_broken_timezone_fails_before_network(cli, fake_tvmaze):
    code, _, err = cli("--tz", "Mars/Olympus", "list")
    assert code == EXIT_ERROR
    assert "Mars/Olympus" in err
    assert fake_tvmaze.calls == []

NOW = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_now(monkeypatch) -> None:
    monkeypatch.setattr("series_watcher.service.utcnow", lambda: NOW)


def test_check_lists_due_episodes_in_user_timezone(cli, fixed_now):
    cli("add", "--id", "75194")

    code, moscow, _ = cli("--tz", "Europe/Moscow", "check")
    _, new_york, _ = cli("--tz", "America/New_York", "check")

    assert code == EXIT_OK
    assert "Пора сообщить:" in moscow
    assert "S2026E193 «Episode 392» — 19.10.2026 21:00" in moscow
    assert "S2026E193 «Episode 392» — 19.10.2026 14:00" in new_york
    assert "новых: 0" in moscow


def test_check_on_empty_list_has_nothing_to_say(cli, fixed_now):
    code, out, _ = cli("check")
    assert code == EXIT_OK
    assert "сериалов: 0" in out
    assert "нечего" in out


def test_check_fails_when_a_show_cannot_be_checked(cli, fake_tvmaze, fixed_now):
    cli("add", "--id", "75194")
    cli("add", "--id", "44933")
    fake_tvmaze.add("/shows/44933/episodes", {"error": "gone"}, status=404)

    code, out, _ = cli("check")
    assert code == EXIT_ERROR
    assert "Severance: не проверился" in out
    assert "Beyond the Gates" in out


def test_check_lookback_narrows_the_window(cli, fixed_now):
    cli("add", "--id", "75194")

    _, wide, _ = cli("--tz", "UTC", "check", "--lookback", "240")
    _, narrow, _ = cli("--tz", "UTC", "check", "--lookback", "24")

    assert wide.count("S2026E") >= narrow.count("S2026E")
    assert "S2026E191" in wide
    assert "S2026E191" not in narrow


def test_run_without_token_says_so(cli, fixed_now):
    cli("add", "--id", "75194")

    code, out, _ = cli("--tz", "Europe/Moscow", "run", "--lookback", "72", "--now", "2026-10-21T12:00:00+00:00")

    assert code == EXIT_OK
    assert "TELEGRAM_BOT_TOKEN" in out
    assert "Отправлено сообщений" not in out


def test_run_dry_run_prints_messages(cli, fixed_now):
    cli("add", "--id", "75194")

    code, out, _ = cli(
        "--tz", "Europe/Moscow", "run", "--dry-run", "--lookback", "72",
        "--now", "2026-10-21T12:00:00+00:00",
    )

    assert code == EXIT_OK
    assert "S2026E193" in out
    assert "Вышел 20.10.2026 21:00" in out
    assert "tvmaze.com" in out


def test_run_in_quiet_hours_is_deferred(cli, fixed_now):
    cli("add", "--id", "75194")

    code, out, _ = cli("--tz", "Europe/Moscow", "run", "--now", "2026-10-19T21:30:00+00:00")

    assert code == EXIT_OK
    assert "Тихий режим (23:00-08:00 Europe/Moscow)" in out
    assert "отложено" in out


def test_digest_is_printed_by_default(cli, fixed_now):
    cli("add", "--id", "75194")

    code, out, _ = cli("--tz", "Europe/Moscow", "digest", "--days", "3")

    assert code == EXIT_OK
    assert "Ближайшие 3 дней" in out
    assert "S2026E196" in out
    assert "tvmaze.com" in out


def test_digest_send_without_token_fails(cli, fixed_now):
    cli("add", "--id", "75194")

    code, _, err = cli("digest", "--send")

    assert code == EXIT_ERROR
    assert "не отправлен" in err


def test_run_rejects_broken_now(cli, fake_tvmaze):
    code, _, err = cli("run", "--now", "вчера вечером")
    assert code == EXIT_ERROR
    assert "ISO 8601" in err


def test_bot_without_token_reports_plainly(cli):
    code, _, err = cli("bot", "--once")
    assert code == EXIT_ERROR
    assert "TELEGRAM_BOT_TOKEN" in err


def test_recommend_on_empty_list_says_what_to_do(cli):
    code, out, _ = cli("recommend")
    assert code == EXIT_OK
    assert "Список пуст" in out


def test_recommend_lists_candidates_with_reasons(cli, fake_tvmaze):
    from series_watcher.recommend import _upcoming

    fake_tvmaze.add("/shows/44933/crew", load_fixture("crew_show"))
    fake_tvmaze.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    for day in _upcoming(3):
        fake_tvmaze.add(f"/schedule?date={day}", load_fixture("schedule"))
    cli("add", "--id", "44933")

    code, out, _ = cli("recommend", "--limit", "3", "--no-tmdb")

    assert code == EXIT_OK
    assert "Похожее на ваш список" in out
    assert "--reject" in out


def test_reject_is_remembered(cli):
    cli("add", "--id", "44933")

    code, out, _ = cli("recommend", "--reject", "777")

    assert code == EXIT_OK
    assert "больше не предлагаю" in out
    assert "not_interested" in out
