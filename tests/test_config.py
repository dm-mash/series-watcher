"""Настройки: ключи только из окружения, зона времени, деградация без ключей."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from series_watcher.config import Config, ConfigError

KEY_NAMES = ("telegram_bot_token", "telegram_chat_id")


def test_service_works_without_any_key():
    config = Config.from_env(env_file=None, environ={})
    assert config.missing_keys() == list(KEY_NAMES)
    assert len(config.degradation_notes()) == 2
    assert config.telegram_bot_token is None
    assert config.telegram_chat_id is None


def test_degradation_is_reported_plainly():
    notes = Config.from_env(env_file=None, environ={}).degradation_notes()
    joined = " ".join(notes)
    assert "TELEGRAM_BOT_TOKEN" in joined
    assert "TELEGRAM_CHAT_ID" in joined


def test_all_keys_present_means_no_degradation():
    config = Config.from_env(
        env_file=None, environ={"TELEGRAM_BOT_TOKEN": "b", "TELEGRAM_CHAT_ID": "42"}
    )
    assert config.missing_keys() == []
    assert config.degradation_notes() == []
    assert config.telegram_chat_id_int == 42


def test_blank_key_counts_as_missing():
    config = Config.from_env(env_file=None, environ={"TELEGRAM_BOT_TOKEN": "   "})
    assert "telegram_bot_token" in config.missing_keys()


def test_env_values_are_parsed():
    config = Config.from_env(
        env_file=None,
        environ={
            "TZ": "Europe/Moscow",
            "DB_PATH": "/tmp/series.db",
            "HTTP_TIMEOUT": "5",
            "HTTP_RETRIES": "1",
            "TVMAZE_CACHE_TTL": "60",
            "TVMAZE_RATE_LIMIT": "10/5",
        },
    )
    assert config.tz_name == "Europe/Moscow"
    assert str(config.tz) == "Europe/Moscow"
    assert config.db_path == Path("/tmp/series.db")
    assert config.http_timeout == 5.0
    assert config.http_retries == 1
    assert config.tvmaze_cache_ttl == 60.0
    assert config.tvmaze_rate_limit == (10, 5.0)


def test_broken_timezone_is_reported():
    with pytest.raises(ConfigError, match="Europe/Moscow"):
        Config.from_env(env_file=None, environ={"TZ": "Mars/Olympus"})


def test_broken_number_is_reported():
    with pytest.raises(ConfigError, match="HTTP_TIMEOUT"):
        Config.from_env(env_file=None, environ={"HTTP_TIMEOUT": "быстро"})


def test_broken_rate_limit_is_reported():
    with pytest.raises(ConfigError, match="TVMAZE_RATE_LIMIT"):
        Config.from_env(env_file=None, environ={"TVMAZE_RATE_LIMIT": "много"})


def test_chat_id_must_be_numeric():
    with pytest.raises(ConfigError, match="TELEGRAM_CHAT_ID"):
        Config.from_env(env_file=None, environ={"TELEGRAM_CHAT_ID": "my channel"}).telegram_chat_id_int


def test_replace_rejects_unknown_setting():
    with pytest.raises(ConfigError, match="неизвестные настройки"):
        Config().replace(db_path="ok", telegram_bot_toke="typo")


def test_env_file_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TZ=Europe/Kaliningrad\nTELEGRAM_BOT_TOKEN=local-token\n", encoding="utf-8")
    try:
        config = Config.from_env(env_file=env_file, environ={})
        assert config.tz_name == "Europe/Kaliningrad"
        assert config.telegram_bot_token == "local-token"
    finally:
        for name in ("TZ", "TELEGRAM_BOT_TOKEN"):
            os.environ.pop(name, None)