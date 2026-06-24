"""
VoC-агент с инструментами: поиск, статистика, топ проблем.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm_client import get_model, make_raw_client
from rag import ask as rag_ask

BASE = Path(__file__).resolve().parent
REVIEWS_JSON = BASE / "output" / "reviews.json"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Семантический поиск по отзывам (RAG). Для вопросов «что писали про X».",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats_by_platform",
            "description": "Средний рейтинг и число отзывов по платформам.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_issues",
            "description": "Топ категорий проблем по severity.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 5}},
            },
        },
    },
]

SYSTEM = """Ты VoC-аналитик приложения NovaPay. Числа — только из инструментов.
Инструменты: search_reviews, stats_by_platform, top_issues.
Ответ — 1-3 предложения на русском."""


def _load_reviews() -> list[dict]:
    if not REVIEWS_JSON.exists():
        return []
    return json.loads(REVIEWS_JSON.read_text(encoding="utf-8"))


def search_reviews(query: str) -> dict:
    ans, path = rag_ask(query)
    return {"answer": ans.answer, "sources": ans.sources, "quotes": ans.quotes}


def stats_by_platform() -> dict:
    reviews = _load_reviews()
    by_plat: dict[str, list[int]] = {}
    for r in reviews:
        p = r.get("platform", "?")
        if r.get("rating"):
            by_plat.setdefault(p, []).append(r["rating"])
    return {
        plat: {"count": len(rs), "avg_rating": round(sum(rs) / len(rs), 2)}
        for plat, rs in by_plat.items()
    }


def top_issues(limit: int = 5) -> dict:
    reviews = _load_reviews()
    scores: Counter[str] = Counter()
    for r in reviews:
        for iss in r.get("issues", []):
            scores[iss["category"]] += iss.get("severity", 1)
    return {"top": scores.most_common(limit)}


TOOLS_IMPL = {
    "search_reviews": search_reviews,
    "stats_by_platform": stats_by_platform,
    "top_issues": top_issues,
}

_PRICE_IN, _PRICE_OUT = 0.27, 1.10


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {"prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0}


def _estimate_cost(u: dict) -> float:
    return u["prompt_tokens"] * _PRICE_IN / 1e6 + u["completion_tokens"] * _PRICE_OUT / 1e6


def run_agent(query: str, *, max_iter: int = 6, verbose: bool = False) -> dict:
    client = make_raw_client()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    trace = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=get_model(), messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.0
        )
        u = _usage(resp)
        usage["prompt_tokens"] += u["prompt_tokens"]
        usage["completion_tokens"] += u["completion_tokens"]
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            trace.append({"step": step, "final": msg.content})
            return {
                "answer": msg.content,
                "trace": trace,
                "steps": step,
                "tools": [t["call"] for t in trace if "call" in t],
                "usage": usage,
                "cost_usd": round(_estimate_cost(usage), 6),
            }
        for tc in msg.tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments or "{}")
            fn = TOOLS_IMPL.get(tc.function.name)
            obs = fn(**args) if fn else {"error": f"unknown {tc.function.name}"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _json.dumps(obs, ensure_ascii=False)})
            trace.append({"step": step, "call": tc.function.name, "args": args, "obs": obs})
            if verbose:
                print(f"  {tc.function.name} -> {str(obs)[:100]}")
    return {
        "answer": None,
        "error": "max_iter",
        "trace": trace,
        "steps": max_iter,
        "tools": [],
        "usage": usage,
        "cost_usd": round(_estimate_cost(usage), 6),
    }
