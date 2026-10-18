"""JustWatch: где сериал сейчас стримят.

Без `JUSTWATCH_API_KEY` источник недоступен, и об этом говорится прямо. Форма
ответа проверена без ключа лишь частично (см. PLAN.md): запросы к API отвечают
422 с текстом ошибки валидации, а `searchTitles` отдаёт `edges`. Разбор
написан терпимо: неизвестное поле — это пустой список предложений, а не падение.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..db import Show
from .base import Offer, SourceError

ENDPOINT = "https://apis.justwatch.com/graphql"

SEARCH_QUERY = """
query SearchTitles($source: String!, $country: String!) {
  searchTitles(source: $source, country: $country) {
    edges { node { id } }
  }
}
"""

OFFERS_QUERY = """
query TitleOffers($id: ID!, $country: String!) {
  nodes(ids: [$id]) {
    __typename
    ... on Show {
      offers {
        monetizationType
        url { standardWebUrl }
      }
    }
  }
}
"""

MONETIZATION = {
    "FLATRATE": "подписка",
    "FREE": "бесплатно",
    "ADS": "с рекламой",
    "RENT": "в аренду",
    "BUY": "в собственность",
}


class JustWatchSource:
    name = "justwatch"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        country: str = "RU",
        transport: Any = None,
        timeout: float = 15.0,
        user_agent: str = "series-watcher/0.1",
    ) -> None:
        import httpx

        self._api_key = api_key or None
        self._country = country
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Content-Type": "application/json", "User-Agent": user_agent},
        )

    def available(self) -> bool:
        return bool(self._api_key)

    def where_to_watch(self, show: Show) -> Sequence[Offer]:
        if not self.available():
            raise SourceError("нет JUSTWATCH_API_KEY")
        title_ids = self._search(show.name)
        if not title_ids:
            return []
        return self._offers(title_ids[0])

    def close(self) -> None:
        self._client.close()

    def _search(self, name: str) -> list[str]:
        data = self._graphql(SEARCH_QUERY, {"source": name, "country": self._country})
        edges = (data.get("searchTitles") or {}).get("edges") or []
        ids: list[str] = []
        for edge in edges:
            node = (edge or {}).get("node") or {}
            value = node.get("id")
            if isinstance(value, str) and value:
                ids.append(value)
        return ids

    def _offers(self, title_id: str) -> list[Offer]:
        data = self._graphql(OFFERS_QUERY, {"id": title_id, "country": self._country})
        nodes = data.get("nodes") or []
        offers: list[Offer] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            for item in node.get("offers") or ():
                url = ((item or {}).get("url") or {}).get("standardWebUrl")
                if not isinstance(url, str) or not url:
                    continue
                kind_raw = item.get("monetizationType")
                kind = MONETIZATION.get(kind_raw, kind_raw) if isinstance(kind_raw, str) else None
                offers.append(Offer(provider="JustWatch", url=url, kind=kind))
        return offers

    def _graphql(self, query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {"X-Api-Key": self._api_key} if self._api_key else {}
        try:
            response = self._client.post(
                ENDPOINT,
                json={"query": query, "variables": dict(variables)},
                headers=headers,
            )
        except Exception as exc:  # noqa: BLE001 - сеть любой формы
            raise SourceError(f"JustWatch не ответил: {exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise SourceError(f"JustWatch вернул не JSON: HTTP {response.status_code}") from exc

        errors = body.get("errors") if isinstance(body, dict) else None
        if response.status_code >= 400 or errors:
            detail = _error_text(errors) or f"HTTP {response.status_code}"
            raise SourceError(f"JustWatch: {detail}")
        data = body.get("data") or {}
        return data if isinstance(data, dict) else {}


def _error_text(errors: Any) -> str | None:
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if isinstance(first, Mapping):
        message = first.get("message")
        return message if isinstance(message, str) else None
    return str(first)