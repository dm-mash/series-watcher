"""Источник «свой»: ссылки из локального файла.

Файл — обычный JSON, который правит пользователь:

```json
{
  "44933": [{"provider": "Кинопоиск", "url": "https://example.tv/s/severance",
             "kind": "subscription"}]
}
```

Ключ — id TVmaze (`series-watcher add --id` его показывает). Файла нет — источник
просто недоступен, и об этом сообщается, а не выдумываются ссылки.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..db import Show
from .base import Offer, SourceError

DEFAULT_PATH = Path("data/where_to_watch.json")


class CustomSource:
    name = "свой"

    def __init__(self, path: Path | str = DEFAULT_PATH, *, clock=None) -> None:
        self.path = Path(path)
        self._clock = clock

    def available(self) -> bool:
        return self.path.exists()

    def add(
        self, tvmaze_id: int, url: str, *, provider: str = "ссылка", kind: str | None = None
    ) -> dict[str, Any]:
        """Дописать ссылку, сохранив всё, что там уже было."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, Mapping):
            raw = {}
        key = str(tvmaze_id)
        offers = [item for item in (raw.get(key) or []) if isinstance(item, (Mapping, str))]
        offers.append({"provider": provider, "url": url, **({"kind": kind} if kind else {})})
        raw[key] = offers
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"provider": provider, "url": url, **({"kind": kind} if kind else {})}

    def where_to_watch(self, show: Show) -> Sequence[Offer]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceError(f"не читается {self.path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise SourceError(f"{self.path}: ожидался объект с id сериала")
        return _parse_offers(raw.get(str(show.tvmaze_id)))


def _parse_offers(value: Any) -> list[Offer]:
    if not isinstance(value, list):
        return []
    offers: list[Offer] = []
    for item in value:
        if isinstance(item, str):
            offers.append(Offer(provider="ссылка", url=item))
            continue
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        offers.append(
            Offer(
                provider=str(item.get("provider") or "ссылка"),
                url=url,
                kind=_as_str(item.get("kind")),
                price=_as_str(item.get("price")),
            )
        )
    return offers


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None