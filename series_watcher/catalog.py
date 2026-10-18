"""Модель там, где она действительно нужна: каталог и формулировки.

Четыре вещи, которые человек делает словами, а поиск по подстроке не умеет:

1. описание сериала -> название, по которому искать в каталоге TVmaze;
2. выбор из кандидатов, когда запрос неоднозначный;
3. перевод и сжатие описания эпизода до одной-двух фраз;
4. объяснение рекомендации одной фразой по фактам из ответа API.

Ни одна из них не обязательна: без модели (`None` или ошибка) вызывающий код
получает `None` и работает дальше как раньше. Факты модель не выдумывает —
ей дают то, что пришло из API.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .llm import LLMClient, LLMError
from .tvmaze import TvMazeShow

TITLE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "название сериала по-английски, как оно есть в каталоге; "
            "пустая строка, если описания недостаточно",
        }
    },
    "required": ["title"],
    "additionalProperties": False,
}

PICK_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": ["integer", "null"],
            "description": "id выбранного сериала из списка; null, если ни один не подходит",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["id", "confidence"],
    "additionalProperties": False,
}

TITLE_PROMPT = (
    "Пользователь описал сериал своими словами. Восстанови его название так, как оно "
    "записано в англоязычном каталоге, и верни только название. Если по описанию "
    "нельзя уверенно понять, какой это сериал, верни пустую строку: лучше пусто, "
    "чем выдуманное название."
)

PICK_PROMPT = (
    "Пользователь хочет добавить сериал. Перед тобой результаты поиска по каталогу. "
    "Выбери тот, о котором говорит пользователь. Если ни один не подходит, верни id=null."
)

SUMMARY_PROMPT = (
    "Сожми описание эпизода до одной-двух фраз на русском и без спойлеров финала. "
    "Не добавляй того, чего в описании нет. Если описания нет, верни пустую строку."
)

ONELINER_PROMPT = (
    "Дай одно предложение на русском о том, о чём этот сериал. Это перевод и сжатие "
    "описания, а не пересказ сюжета и не перечисление героев.\n"
    "Правила:\n"
    "- РОВНО ОДНО предложение, короче 20 слов;\n"
    "- без спойлеров, без имён персонажей, без оценочных прилагательных;\n"
    "- если описания нет, верни пустую строку."
)

EXPLAIN_PROMPT = (
    "Ты объясняешь, почему сериал рекомендуют человеку, который уже смотрит другое.\n"
    "Правила:\n"
    "- приведи РОВНО ОДНО предложение, короче 15 слов;\n"
    "- название сериала и жанры не повторяй — они уже напечатаны отдельно;\n"
    "- бери факт из списка причин и ничего больше: не пиши «известный», «в похожем "
    "стиле», «драматические проекты» и прочее, чего в причинах нет;\n"
    "- отвечай кратко и сухо, как подпись к фото.\n"
    "Пример причины «продюсер: Graham Yost» → «Тот же продюсер, Graham Yost.»\n"
    "Причина «жанры: Drama, Comedy» → «С теми же жанрами: Drama, Comedy.»"
)


def _client(llm: LLMClient | None) -> LLMClient | None:
    return llm if llm is not None and llm.configured else None


def title_from_description(description: str, *, llm: LLMClient | None) -> str | None:
    """«Сериал, где людям режут память» -> название для поиска по каталогу."""
    client = _client(llm)
    if client is None or not description.strip():
        return None
    try:
        data = client.complete_json(
            [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": description},
            ],
            TITLE_SCHEMA,
            name="title",
        )
    except LLMError:
        return None
    if not isinstance(data, Mapping):
        return None
    title = data.get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


def pick_show(
    description: str,
    candidates: Sequence[TvMazeShow],
    *,
    llm: LLMClient | None,
    floor: float = 0.6,
) -> TvMazeShow | None:
    """Выбрать сериал из выдачи поиска, когда запрос неоднозначный."""
    client = _client(llm)
    if client is None or not candidates:
        return None
    listing = "\n".join(
        f"{show.id}. {show.name} ({', '.join(part for part in (show.kind, show.status, show.network) if part)})"
        for show in candidates
    )
    try:
        data = client.complete_json(
            [
                {"role": "system", "content": PICK_PROMPT},
                {"role": "user", "content": f"Запрос: {description}\n\nКандидаты:\n{listing}"},
            ],
            PICK_SCHEMA,
            name="pick",
        )
    except LLMError:
        return None
    if not isinstance(data, Mapping):
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    chosen = data.get("id")
    if not isinstance(chosen, int) or confidence < floor:
        return None
    return next((show for show in candidates if show.id == chosen), None)


def short_summary(text: str | None, *, llm: LLMClient | None) -> str | None:
    """Описание эпизода на русском, коротко. Без модели — `None`."""
    client = _client(llm)
    if client is None or not text or not text.strip():
        return None
    try:
        reply = client.complete(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text},
            ]
        )
    except LLMError:
        return None
    cleaned = reply.text.strip().split("\n")[0].strip()
    return cleaned or None


def one_liner(title: str, summary: str | None, *, llm: LLMClient | None) -> str | None:
    """Одно предложение о самом сериале: перевод и сжатие описания TVmaze.

    Это не «почему он вам подходит» — этим занимается `explain`.
    """
    client = _client(llm)
    if client is None or not summary or not summary.strip():
        return None
    try:
        reply = client.complete(
            [
                {"role": "system", "content": ONELINER_PROMPT},
                {"role": "user", "content": f"Сериал: {title}\nОписание: {summary}"},
            ]
        )
    except LLMError:
        return None
    return _one_sentence(reply.text, limit=160)


def explain(title: str, reasons: Sequence[str], *, llm: LLMClient | None) -> str | None:
    """Одна фраза «почему это вам подойдёт» — только из перечисленных причин.

    Лимит токенов намеренно общий (`LLM_MAX_TOKENS`): у reasoning-моделей вроде
    gpt-oss часть бюджета уходит на размышление, и небольшой лимит обрывает
    ответ раньше, чем появится текст.
    """
    client = _client(llm)
    if client is None or not reasons:
        return None
    joined = "; ".join(reasons)
    try:
        reply = client.complete(
            [
                {"role": "system", "content": EXPLAIN_PROMPT},
                {"role": "user", "content": f"Сериал: {title}\nПричины: {joined}"},
            ]
        )
    except LLMError:
        return None
    return _one_sentence(reply.text)


def _one_sentence(text: str, limit: int = 140) -> str | None:
    """Первое предложение и не длиннее лимита: краткость тут не украшение."""
    cleaned = " ".join(text.split())
    for stop in (". ", "! ", "? "):
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]
            break
    cleaned = cleaned.rstrip(" .!?,;:")
    if not cleaned:
        return None
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rsplit(" ", 1)[0]
    return cleaned