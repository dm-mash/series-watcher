"""LLM-клиент: ретраи, разбор JSON по схеме, деградация без настройки."""

from __future__ import annotations

import json

import pytest

from series_watcher.config import Config
from series_watcher.llm import LLMClient, LLMError, LLMNotConfigured, LLMTruncated


class FakeLLM:
    """Подмена /chat/completions: отдаёт заготовленные ответы по очереди."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def transport(self):
        import httpx

        queue = list(self.responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode("utf-8")))
            self.headers.append(dict(request.headers))
            item = queue.pop(0) if queue else {"choices": [{"message": {"content": "{}"}}]}
            if isinstance(item, int):
                return httpx.Response(item, text="upstream is unhappy")
            return httpx.Response(200, json=item)

        return httpx.MockTransport(handler)

    @property
    def content(self) -> str:
        return self.requests[0]["messages"][-1]["content"]


def reply(content: str, *, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def client(fake: FakeLLM, sleeps: list[float], **env) -> LLMClient:
    config = Config.from_env(
        env_file=None,
        environ={"LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "model-1", **env},
    )
    return LLMClient(config, transport=fake.transport(), sleep=sleeps.append)


def test_without_settings_it_refuses_to_guess():
    unconfigured = LLMClient(Config())
    assert unconfigured.configured is False
    with pytest.raises(LLMNotConfigured):
        unconfigured.complete([{"role": "user", "content": "привет"}])


def test_complete_sends_model_and_messages(sleeps):
    fake = FakeLLM(reply("привет"))
    result = client(fake, sleeps).complete([{"role": "user", "content": "привет"}])

    assert result.text == "привет"
    assert fake.requests[0]["model"] == "model-1"
    assert fake.requests[0]["messages"][-1]["content"] == "привет"


def test_api_key_goes_into_authorization_header(sleeps):
    fake = FakeLLM(reply("привет"))
    client(fake, sleeps, LLM_API_KEY="secret").complete([{"role": "user", "content": "x"}])

    assert fake.headers[0]["authorization"] == "Bearer secret"
    assert "api_key" not in fake.requests[0]


def test_json_answer_follows_the_schema(sleeps):
    fake = FakeLLM(reply('{"intent": "digest", "confidence": 0.9}'))
    schema = {"type": "object", "properties": {"intent": {"type": "string"}}}

    data = client(fake, sleeps).complete_json([{"role": "user", "content": "дайджест"}], schema)

    assert data == {"intent": "digest", "confidence": 0.9}
    response_format = fake.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema


def test_json_in_code_fence_is_still_parsed(sleeps):
    fake = FakeLLM(reply('```json\n{"intent": "digest"}\n```'))
    assert client(fake, sleeps).complete_json([], {}) == {"intent": "digest"}


def test_transient_failure_is_retried(sleeps):
    fake = FakeLLM(503, 503, reply("готово"))

    assert client(fake, sleeps).complete([{"role": "user", "content": "x"}]).text == "готово"
    assert len(fake.requests) == 3
    assert len(sleeps) == 2


def test_giving_up_after_retries(sleeps):
    fake = FakeLLM(500, 500, 500, 500)

    with pytest.raises(LLMError):
        client(fake, sleeps, HTTP_RETRIES="2").complete([{"role": "user", "content": "x"}])

    assert len(fake.requests) == 3


def test_truncated_answer_is_reported(sleeps):
    fake = FakeLLM(reply("начало...", finish="length"))

    with pytest.raises(LLMTruncated):
        client(fake, sleeps).complete([{"role": "user", "content": "x"}])


def test_provider_error_is_surfaced(sleeps):
    fake = FakeLLM({"error": {"message": "model not found"}})

    with pytest.raises(LLMError, match="model not found"):
        client(fake, sleeps).complete([{"role": "user", "content": "x"}])


def test_non_json_answer_is_reported(sleeps):
    fake = FakeLLM(reply("просто текст"))

    with pytest.raises(LLMError, match="не JSON"):
        client(fake, sleeps).complete_json([], {})