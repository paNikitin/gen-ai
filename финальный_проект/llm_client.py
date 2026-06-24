"""OpenAI-совместимый клиент со structured outputs."""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
T = TypeVar("T")


def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Задай LLM_AUTH_TOKEN в .env")
        http = httpx.Client(verify=False, timeout=float(os.environ.get("LLM_TIMEOUT", "200")))
        return OpenAI(api_key=key, base_url=base, http_client=http)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Задай LLM_AUTH_TOKEN или OPENAI_API_KEY")
    return OpenAI(api_key=key)


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def _thinking_off_payload() -> dict:
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "reasoning_effort": "none"}


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str):
    t = _clean(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Нет JSON в ответе: {text[:300]!r}")


class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 2,
        temperature: float = 0.0,
        with_completion: bool = False,
        **kw: Any,
    ):
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            schema = {"type": "object", "properties": {"items": {"type": "array", "items": TypeAdapter(item_type).json_schema()}}, "required": ["items"]}
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()
        addendum = f"\n\nJSON по схеме:\n{json.dumps(schema, ensure_ascii=False)}\nТОЛЬКО JSON."
        if wrap_list:
            addendum += " Массив в поле items."
        msgs = [dict(m) for m in messages]
        si = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if si is not None:
            msgs[si]["content"] += addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum})
        thinking = _thinking_off_payload()
        last_err = None
        raw = ""
        for _ in range(max_retries + 1):
            try:
                try:
                    resp = self._c.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"}, temperature=temperature, **thinking)
                except TypeError:
                    resp = self._c.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"}, temperature=temperature)
                except Exception as sdk_err:
                    msg = str(sdk_err)
                    if ("reasoning_effort" in msg or "chat_template" in msg) and thinking:
                        thinking = {}
                        resp = self._c.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"}, temperature=temperature)
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                result = adapter.validate_python(obj)
                return (result, resp) if with_completion else result
            except Exception as e:
                last_err = e
                msgs.extend([{"role": "assistant", "content": raw}, {"role": "user", "content": f"Ошибка: {e}. Верни валидный JSON."}])
        raise last_err


class _Chat:
    def __init__(self, c: OpenAI):
        self.completions = _Completions(c)


class JsonClient:
    def __init__(self, c: OpenAI):
        self.chat = _Chat(c)


def make_client() -> JsonClient:
    return JsonClient(_make_openai_client())


def make_raw_client() -> OpenAI:
    """Сырой клиент без reasoning_effort (совместимость с DeepSeek)."""
    return _make_openai_client()


def check_api() -> tuple[bool, str]:
    """Проверка доступности LLM API (один короткий запрос)."""
    try:
        resp = make_raw_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=2,
            temperature=0.0,
        )
        if resp.choices[0].message.content is not None:
            return True, "ok"
        return True, "ok"
    except Exception as e:
        return False, str(e)
