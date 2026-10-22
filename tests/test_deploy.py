"""Юниты systemd: проверяем то, что легко сломать молча."""

from __future__ import annotations

from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
PROJECT = Path(__file__).resolve().parent.parent
EXECUTABLE = PROJECT / ".venv/bin/series-watcher"


def _unit(name: str) -> str:
    return (DEPLOY / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    ["series-watcher.service", "series-watcher-weekly.service", "series-watcher-bot.service"],
)
def test_service_runs_as_the_project_owner(name):
    """Без `User=` сервис пошёл бы от root и писал бы в базу от него же."""
    assert "User=${USER}" in _unit(name)


@pytest.mark.parametrize("name", ["series-watcher.service", "series-watcher-weekly.service"])
def test_oneshot_units_do_not_pin_the_timezone(name):
    """`TZ=` в юните перебивает `.env`: таймер показывал бы время не как бот."""
    assert "Environment=TZ=" not in _unit(name)


def test_bot_survives_failure_and_freshes():
    """Бот держит long polling часами: упасть и не подняться он не должен."""
    unit = _unit("series-watcher-bot.service")
    assert "Type=simple" in unit
    assert "Restart=always" in unit
    assert "RestartSec=" in unit


def test_timer_polls_no_often_than_hourly():
    assert "OnUnitActiveSec=1h" in _unit("series-watcher.timer")


@pytest.mark.parametrize(
    "name",
    ["series-watcher.service", "series-watcher-weekly.service", "series-watcher-bot.service"],
)
def test_exec_start_points_at_the_installed_script(name):
    """Путь в ExecStart должен существовать, иначе сервис не стартует вовсе."""
    assert "ExecStart=${PROJECT_DIR}/.venv/bin/series-watcher " in _unit(name)


def test_rendered_unit_points_at_existing_script(tmp_path, monkeypatch):
    """После подстановки переменных ExecStart указывает на реальный скрипт."""
    import os
    import subprocess

    env = {**os.environ, "USER": "smash", "PROJECT_DIR": str(PROJECT)}
    rendered = subprocess.run(
        ["envsubst", "$USER $PROJECT_DIR"],
        input=_unit("series-watcher.service"),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert f"ExecStart={EXECUTABLE} " in rendered
