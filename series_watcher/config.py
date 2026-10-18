"""Настройки из окружения.

Ключи читаются только из окружения (`.env` рядом с проектом или переменные
системы). Ни одного ключа в репозитории быть не должно, и сервис обязан
работать без каждого из них: `missing_keys()` и `degradation_notes()` дают
список того, что отключено, чтобы об этом можно было сказать прямо.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values, load_dotenv

DEFAULT_TZ = "UTC"
DEFAULT_DB_PATH = Path("data/series-watcher.db")
DEFAULT_BASE_URL = "https://api.tvmaze.com"
DEFAULT_USER_AGENT = "series-watcher/0.1 (+https://github.com/series-watcher)"

#: Лимит TVmaze — 20 запросов на 10 секунд на IP, берём с запасом.
DEFAULT_RATE_LIMIT = (18, 10.0)

#: Ключи, без которых возможности выключаются, а не падают.
KEY_FIELDS = (
    "tmdb_api_key",
    "justwatch_api_key",
    "telegram_bot_token",
    "telegram_chat_id",
)

DEGRADATION_NOTES = {
    "tmdb_api_key": (
        "нет TMDB_API_KEY: рекомендации по режиссёрам и сценаристам "
        "недоступны, остаются авторство из TVmaze и жанры"
    ),
    "justwatch_api_key": (
        "нет JUSTWATCH_API_KEY: «где смотреть» отдаём только ссылку на TVmaze"
    ),
    "telegram_bot_token": (
        "нет TELEGRAM_BOT_TOKEN: уведомления не отправляются, работает только CLI"
    ),
    "telegram_chat_id": (
        "нет TELEGRAM_CHAT_ID: некуда слать уведомления, работает только CLI"
    ),
}


class ConfigError(ValueError):
    """Неверное значение в окружении."""


@dataclass(frozen=True)
class Config:
    """Значения по умолчанию рассчитаны на работу без единого ключа."""

    tz_name: str = DEFAULT_TZ
    db_path: Path = DEFAULT_DB_PATH
    http_timeout: float = 15.0
    http_retries: int = 3
    user_agent: str = DEFAULT_USER_AGENT
    tvmaze_base_url: str = DEFAULT_BASE_URL
    tvmaze_cache_ttl: float = 21600.0
    tvmaze_rate_limit: tuple[int, float] = DEFAULT_RATE_LIMIT
    tmdb_api_key: str | None = None
    justwatch_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    def __post_init__(self) -> None:
        _check_tz(self.tz_name)
        object.__setattr__(self, "db_path", Path(self.db_path))

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)

    @property
    def telegram_chat_id_int(self) -> int | None:
        if not self.telegram_chat_id:
            return None
        try:
            return int(self.telegram_chat_id)
        except ValueError as exc:
            raise ConfigError(
                f"TELEGRAM_CHAT_ID должен быть числом, получено {self.telegram_chat_id!r}"
            ) from exc

    def missing_keys(self) -> list[str]:
        return [name for name in KEY_FIELDS if not getattr(self, name)]

    def degradation_notes(self) -> list[str]:
        return [DEGRADATION_NOTES[name] for name in self.missing_keys()]

    @classmethod
    def from_env(
        cls,
        env_file: str | os.PathLike[str] | None = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> "Config":
        env: Mapping[str, str]
        if environ is None:
            # Системное окружение важнее файла: ключи не должны молча
            # подменяться значениями из `.env`.
            if env_file is not None:
                load_dotenv(env_file, override=False)
            env = os.environ
        else:
            from_file = dict(dotenv_values(env_file)) if env_file is not None else {}
            env = {name: value for name, value in from_file.items() if value is not None}
            env.update(environ)

        db_path = Path(env.get("DB_PATH") or DEFAULT_DB_PATH)
        rate_max, rate_window = DEFAULT_RATE_LIMIT
        raw_limit = env.get("TVMAZE_RATE_LIMIT")
        if raw_limit:
            try:
                parts = raw_limit.split("/", 1)
                rate_max = int(parts[0])
                rate_window = float(parts[1]) if len(parts) > 1 else 10.0
            except ValueError as exc:
                raise ConfigError(
                    f"TVMAZE_RATE_LIMIT ожидается как N/секунды, получено {raw_limit!r}"
                ) from exc
            if rate_max <= 0 or rate_window <= 0:
                raise ConfigError(f"TVMAZE_RATE_LIMIT должен быть положительным: {raw_limit!r}")

        config = cls(
            tz_name=env.get("TZ") or _system_tz_name(),
            db_path=db_path,
            http_timeout=_float(env, "HTTP_TIMEOUT", 15.0),
            http_retries=_int(env, "HTTP_RETRIES", 3),
            user_agent=env.get("USER_AGENT") or DEFAULT_USER_AGENT,
            tvmaze_base_url=(env.get("TVMAZE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            tvmaze_cache_ttl=_float(env, "TVMAZE_CACHE_TTL", 21600.0),
            tvmaze_rate_limit=(rate_max, rate_window),
            tmdb_api_key=_opt(env, "TMDB_API_KEY"),
            justwatch_api_key=_opt(env, "JUSTWATCH_API_KEY"),
            telegram_bot_token=_opt(env, "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_opt(env, "TELEGRAM_CHAT_ID"),
            llm_base_url=_opt(env, "LLM_BASE_URL"),
            llm_api_key=_opt(env, "LLM_API_KEY"),
            llm_model=_opt(env, "LLM_MODEL"),
        )
        return config

    def replace(self, **changes: object) -> "Config":
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise ConfigError(f"неизвестные настройки: {sorted(unknown)}")
        return replace(self, **changes)  # type: ignore[arg-type]


def _opt(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть целым числом, получено {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть числом, получено {raw!r}") from exc


def _system_tz_name() -> str:
    """Имя системной зоны, если она известна, иначе UTC."""
    try:
        target = str(Path("/etc/localtime").resolve())
    except OSError:
        return DEFAULT_TZ
    if "zoneinfo/" not in target:
        return DEFAULT_TZ
    name = target.split("zoneinfo/", 1)[1]
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return DEFAULT_TZ
    return name


def _check_tz(name: str) -> None:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise ConfigError(
            f"TZ={name!r} не похоже на IANA-зону, например Europe/Moscow или UTC"
        ) from exc