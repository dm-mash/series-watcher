"""Тонкий LLM-клиент: POST, ретраи, ответ по JSON-схеме.

Говорит с любым OpenAI-совместимым `/chat/completions`. Клиент намеренно свой, а
не переиспользованный из `llm-bot`: там в payload только `model/messages/
temperature/max_tokens`, а разбор фразы в намерение без схемы ненадёжен.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .config import Config

CHAT_ENDPOINT = "/chat/completions"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_SLEEP = 20.0


class LLMError(RuntimeError):
    """Модель не ответила так, как мы просим."""


class LLMNotConfigured(LLMError):
    """Не заданы LLM_BASE_URL или LLM_MODEL."""


class LLMTruncated(LLMError):
    """Ответ оборвался на лимите токенов и разбирать нечего."""


@dataclass(frozen=True)
class Reply:
    text: str
    payload: Mapping[str, Any]


class LLMClient:
    def __init__(
        self,
        config: Config,
        *,
        transport: Any = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float | None = None,
    ) -> None:
        import httpx

        self._config = config
        self._sleep = sleep
        self._retries = max(0, config.http_retries)
        self._client = client or httpx.Client(
            base_url=(config.llm_base_url or "").rstrip("/"),
            timeout=timeout or config.http_timeout,
            transport=transport,
            headers={"Content-Type": "application/json", "User-Agent": config.user_agent},
        )
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return bool(self._config.llm_base_url and self._config.llm_model)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        if not self.configured:
            raise LLMNotConfigured("не заданы LLM_BASE_URL или LLM_MODEL")
        payload: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": [dict(message) for message in messages],
            "temperature": self._config.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._config.llm_max_tokens,
        }
        body = self._post(payload)
        return Reply(text=_content(body, payload), payload=body)

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        *,
        name: str = "result",
        temperature: float | None = None,
    ) -> Any:
        """Ответ строго по схеме: провайдеру отдаётся `response_format`."""
        if not self.configured:
            raise LLMNotConfigured("не заданы LLM_BASE_URL или LLM_MODEL")
        payload: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": [dict(message) for message in messages],
            "temperature": self._config.llm_temperature if temperature is None else temperature,
            "max_tokens": self._config.llm_max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": dict(schema)},
            },
        }
        body = self._post(payload)
        return _parse_json(_content(body, payload))

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        import httpx

        url = f"{str(self._client.base_url).rstrip('/')}{CHAT_ENDPOINT}"
        headers = {}
        if self._config.llm_api_key:
            headers["Authorization"] = f"Bearer {self._config.llm_api_key}"
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = exc
                if attempt == self._retries:
                    break
                self._sleep(min(MAX_SLEEP, 0.5 * (2**attempt)))
                continue
            if response.status_code in RETRY_STATUSES:
                last = LLMError(f"LLM ответил {response.status_code}")
                if attempt == self._retries:
                    break
                self._sleep(min(MAX_SLEEP, 0.5 * (2**attempt)))
                continue
            if response.status_code >= 400:
                raise LLMError(f"LLM ответил {response.status_code}: {response.text[:200]}")
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError("LLM вернул не JSON") from exc
            if isinstance(body, dict) and body.get("error"):
                raise LLMError(f"LLM вернул ошибку: {str(body['error'])[:200]}")
            if not isinstance(body, dict):
                raise LLMError("LLM вернул не объект")
            return body
        raise LLMError(f"LLM не ответил: {last}") from last


def _content(body: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise LLMError("LLM вернул пустой ответ")
    choice = choices[0] or {}
    if choice.get("finish_reason") == "length":
        raise LLMTruncated("ответ оборвался на лимите токенов")
    text = (choice.get("message") or {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("LLM вернул пустой текст")
    return text


def _parse_json(text: str) -> Any:
    """Провайдер может вернуть JSON в «приукрашенном» виде — снимаем обвязку."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except ValueError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except ValueError:
                pass
        raise LLMError(f"LLM вернул не JSON: {text[:120]}") from exc