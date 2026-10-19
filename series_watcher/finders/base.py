"""Договорённости между сервисом и тем, кто ищет раздачи.

Поиск — сменяемая деталь, как источник расписания или рекомендатель: сервис
знает только `EpisodeFinder`, конкретный трекер живёт в реализации. Если
реализации нет или её пакет не установлен — это не поломка, а «искать нечем»,
и об этом говорится прямо.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class FinderError(Exception):
    """Поиск не удался: сеть, трекер или кривой ответ."""


class FinderUnavailable(FinderError):
    """Искать нечем: пакет не установлен или поиск выключен в настройках."""


@dataclass(frozen=True)
class TorrentHit:
    """Одна найденная раздача.

    `page_url` — страница на трекере: её и показываем человеку. Магнит живёт
    отдельно и достаётся по `id`, когда человек его попросил.
    """

    torrent_id: str
    filename: str
    source: str
    seeders: int = 0
    leechers: int = 0
    size: str | None = None
    page_url: str | None = None
    #: Магнет достаётся отдельно и позже: держать его на каждой находке незачем,
    #: трекеры прячут ссылки по умолчанию. Показываем текстом, не ссылкой:
    #: Telegram не делает `magnet:` кликабельным.
    magnet: str | None = None

    @property
    def alive(self) -> bool:
        """Раздача с сидами. Без них можно не показывать: качать некому."""
        return self.seeders > 0


class EpisodeFinder(Protocol):
    """Кто умеет искать раздачу эпизода."""

    name: str
    description: str

    def available(self) -> bool:
        """Готов ли искать. `False` — ищем не будем, и это не ошибка."""

    def search(self, show: str, season: int, number: int, *, limit: int = 5) -> Sequence[TorrentHit]:
        """Раздачи серии `season`/`number` сериала `show`.

        Порядок — по убыванию сидов; плохие (без сидов, не про тот сериал)
        возвращать не нужно.
        """

    def magnet(self, torrent_id: str) -> str | None:
        """Магнит по id раздачи. `None`, если трекер его не отдаёт."""

def prune_magnet(magnet: str | None) -> str | None:
    """Оставить в магнете только хеш и имя.

    Трекеры в ссылке не нужны: клиент качает по infohash, а список аннонсеров
    раздувает ссылку до полутора килобайт. Применяется и к свежим находкам, и к
    сохранённым ранее: старые записи могли лечь в базу до того, как мы стали
    обрезать, и иначе остались бы длинными навсегда.
    """
    if not magnet or not magnet.startswith("magnet:?"):
        return magnet
    kept = [
        item
        for item in magnet[len("magnet:?") :].split("&")
        if item.split("=", 1)[0] in ("xt", "dn")
    ]
    if not kept:
        return magnet
    return "magnet:?" + "&".join(kept)
