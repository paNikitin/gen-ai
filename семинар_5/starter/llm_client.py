"""
Фабрика OpenAI-совместимого клиента + лёгкий JSON-инструктор.

Все параметры берутся из окружения (или .env через python-dotenv).
Поддержан self-signed хост: verify=False, таймаут 200 с.

Ожидаемые переменные окружения:
  LLM_BASE_URL   — базовый URL в духе https://host/v1
  LLM_AUTH_TOKEN — bearer-токен
  LLM_MODEL      — имя модели для chat.completions

Fallback: если LLM_BASE_URL не задан — идём в публичный OpenAI и требуем OPENAI_API_KEY.

Почему тут свой JSON-слой, а не `instructor`
--------------------------------------------
Мы ходим в разные self-hosted эндпоинты (Qwen/vLLM, gpt-oss через hydragpt и т.п.).
У каждого свои причуды:
  * vLLM+xgrammar валится на pydantic-схемах с `$defs` (tool-calling mode).
  * gpt-oss любит досылать после JSON токены harmony вида `<|constrain|>json<|message|>{...}`,
    из-за чего pydantic видит trailing characters и падает.

Нам не нужен весь instructor с его tool-calling/retry-семантикой. Хватает:
  1. Послать запрос с response_format={"type":"json_object"} (общий знаменатель).
  2. Достать из ответа первый валидный JSON-объект или массив, игнорируя мусор.
  3. Сверить с pydantic.
Всё это — 60 строк, зато контроль полный.

wrap_openai_json() оборачивает низкоуровневый OpenAI-клиент в объект с
drop-in API `client.chat.completions.create(..., response_model=...)` — чтобы
вызывающий код не менял каждый вызов.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

# .env загрузим, если есть python-dotenv. find_dotenv ходит вверх по дереву каталогов.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# Глушим InsecureRequestWarning из urllib3 — verify=False намеренно.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


T = TypeVar("T")


def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_AUTH_TOKEN не задан. Либо экспортируй токен, "
                "либо положи LLM_AUTH_TOKEN=... в .env."
            )
        timeout = float(os.environ.get("LLM_TIMEOUT", "200"))
        http = httpx.Client(verify=False, timeout=timeout)
        return OpenAI(api_key=key, base_url=base, http_client=http)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Ни LLM_BASE_URL, ни OPENAI_API_KEY не заданы. "
            "Сконфигурируй стенд через .env (см. .env.example)."
        )
    return OpenAI(api_key=key)


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------------------------
# JSON-парсинг из грязного ответа LLM
# ---------------------------------------------------------------------------

_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def _thinking_off_payload() -> dict:
    """
    Собрать kwargs, которые отключают reasoning-режим на большинстве
    OpenAI-совместимых серверов. Если переменная окружения LLM_THINKING=on —
    возвращаем пустой dict (думай, сколько хочешь).

    Обоснование:
      * Qwen3 / QwQ на vLLM: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
        — штатный способ, документирован Qwen.
      * gpt-oss / SGLang / vLLM: `reasoning_effort="none"` (у SGLang это literal
        из {none, low, medium, high}; у OpenAI gpt-oss допустим и "minimal", но
        "none" приняли обе вселенные, поэтому используем его).
      * Незнакомые поля сервер обычно игнорирует, так что кидаем оба сразу.
    """
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}
    return {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_effort": "none",
    }


def _clean(text: str) -> str:
    """Снять harmony-токены и markdown-обёртку."""
    text = _HARMONY_RE.sub("", text).strip()
    # ```json ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str):
    """Найти и декодировать первый сбалансированный JSON (object|array)."""
    t = _clean(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"В ответе не найдено валидного JSON: {text[:300]!r}")


# ---------------------------------------------------------------------------
# Drop-in обёртка с API, совместимым с instructor
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 1,
        temperature: float = 0.0,
        **kw: Any,
    ) -> T:
        # list[Model] → оборачиваем в {items: [...]}, т.к. JSON mode требует object
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": item_schema}},
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        addendum = (
            f"\n\nОтвечай ОДНИМ валидным JSON-объектом по схеме:\n{schema_str}\n"
            "ТОЛЬКО JSON. Никакого текста до/после, никакого markdown, "
            "никаких комментариев, никаких повторных объектов."
        )
        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum.lstrip()})

        # Отключаем reasoning — иначе Qwen3 может по 30+ секунд «думать» перед ответом.
        thinking_kw = _thinking_off_payload()

        def _call(kw: dict):
            try:
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **kw,
                )
            except TypeError:
                # Старый SDK не знает reasoning_effort
                safe = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **safe,
                )

        last_err: Exception | None = None
        raw: str = ""
        for _ in range(max_retries + 1):
            try:
                try:
                    resp = _call(thinking_kw)
                except Exception as sdk_err:
                    # Сервер не переварил reasoning_effort / extra_body — сбросим их и повторим.
                    msg = str(sdk_err)
                    bad = "reasoning_effort" in msg or "chat_template_kwargs" in msg or "enable_thinking" in msg
                    if bad and thinking_kw:
                        thinking_kw = {}
                        resp = _call(thinking_kw)
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                return adapter.validate_python(obj)
            except Exception as e:
                last_err = e
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({
                    "role": "user",
                    "content": f"Невалидный ответ: {e}. Верни ТОЛЬКО один корректный JSON по схеме.",
                })
        assert last_err is not None
        raise last_err


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    """Drop-in замена instructor-клиента."""

    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _Chat(openai_client)


def make_client() -> JsonClient:
    """Вернуть клиент с API `client.chat.completions.create(..., response_model=...)`."""
    return JsonClient(_make_openai_client())


# ---------------------------------------------------------------------------
# «Сырой» клиент без JSON-инструктора, но с автоотключением reasoning
# ---------------------------------------------------------------------------
# Нам нужен в семинаре 2: мы хотим увидеть грязный ответ модели как есть
# (markdown, «возраст словом», пост-амбула). Но reasoning всё равно надо
# гасить — иначе Qwen3 думает по 30 секунд перед каждым ответом, а никакого
# учебного смысла это не несёт.


class _RawCompletions:
    """Прокси над openai.chat.completions: инжектирует thinking-off kwargs."""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kw: Any):
        thinking = _thinking_off_payload()

        def _call(extra: dict):
            try:
                return self._inner.create(**kw, **extra)
            except TypeError:
                # Старый SDK не знает reasoning_effort — снимаем и повторяем.
                safe = {k: v for k, v in extra.items() if k != "reasoning_effort"}
                return self._inner.create(**kw, **safe)

        try:
            return _call(thinking)
        except Exception as e:
            msg = str(e)
            bad = (
                "reasoning_effort" in msg
                or "chat_template_kwargs" in msg
                or "enable_thinking" in msg
            )
            if bad and thinking:
                # Сервер не переварил — повторим без thinking-kwargs.
                return _call({})
            raise


class _RawChat:
    def __init__(self, inner):
        self.completions = _RawCompletions(inner.completions)


class RawClient:
    """
    Тонкая обёртка над OpenAI-клиентом: интерфейс такой же
    (`client.chat.completions.create(...)`), но каждый вызов автоматически
    получает kwargs «выключи reasoning», с graceful fallback-ом, если сервер
    не узнаёт эти поля.
    """

    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _RawChat(openai_client.chat)

    # Прокси-доступ ко всему остальному (embeddings, models и т.п.), чтобы
    # не мешать тем, кто захочет их вызвать.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def make_raw_client() -> RawClient:
    """
    Вернуть «сырой» клиент без JSON-инструктора, но с выключенным reasoning.
    Нужен, когда хочется увидеть грязный ответ модели как есть (например, в
    учебном «сломанном» скрипте семинара 2, где мы специально ловим markdown,
    «возраст словом» и прочие прелести).
    """
    return RawClient(_make_openai_client())
