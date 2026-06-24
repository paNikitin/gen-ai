"""
Eval: ≥15 тестов — правильность + путь (шаги, инструменты, токены, trace).
Только online: при ошибке API процесс останавливается.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent import run_agent
from llm_client import check_api
from orchestrator import run_multi
from rag import ask as rag_ask, hybrid_retrieve, ingest

BASE = Path(__file__).resolve().parent
RESULTS_OUT = BASE / "output" / "eval_results.json"
TRACE_OUT = BASE / "output" / "eval_trace.jsonl"
BM25_CACHE = BASE / "output" / "bm25_cache.json"

CASES = [
  {"id": 1, "mode": "rag", "query": "Жалобы на переводы по СБП и зависание денег", "must": ["сбп", "завис"], "gold_sources": ["review_1"]},
  {"id": 2, "mode": "rag", "query": "Реклама кредитов и баннеры в приложении", "must": ["реклам", "микроза"], "gold_sources": ["review_3"]},
  {"id": 3, "mode": "rag", "query": "Проблемы с Face ID и входом в приложение", "must": ["face"], "gold_sources": ["review_8"]},
  {"id": 4, "mode": "rag", "query": "Отзывы про дизайн и устаревший интерфейс", "must": ["дизайн"], "gold_sources": ["review_4", "review_37"]},
  {"id": 5, "mode": "rag", "query": "Сколько ждали поддержку и жалобы на чат", "must": ["поддерж"], "gold_sources": ["review_1", "review_11"]},
  {"id": 6, "mode": "rag", "query": "Упоминания конкурентов Тинькофф и Сбер", "must": ["тинькофф"], "gold_sources": ["review_3", "review_6"]},
  {"id": 7, "mode": "rag", "query": "Баги с батареей и расходом в фоне", "must": ["батар"], "gold_sources": ["review_14"]},
  {"id": 8, "mode": "rag", "query": "Проблемы с оплатой ЖКХ в RuStore версии", "must": ["жкх"], "gold_sources": ["review_35"]},
  {"id": 9, "mode": "agent", "query": "Какой средний рейтинг на Google Play?", "must": ["google"], "expect_tools": ["stats_by_platform"]},
  {"id": 10, "mode": "agent", "query": "Какие топ проблемы по категориям?", "must": ["надёжност"], "expect_tools": ["top_issues"]},
  {"id": 11, "mode": "agent", "query": "Что писали про кэшбэк?", "must": ["кэшбэк"], "expect_tools": ["search_reviews"]},
  {"id": 12, "mode": "agent", "query": "Сравни отзывы про поддержку", "must": ["поддерж"], "expect_tools": ["search_reviews"]},
  {"id": 13, "mode": "agent", "query": "Статистика по всем платформам", "must": ["app store"], "expect_tools": ["stats_by_platform"]},
  {"id": 14, "mode": "agent", "query": "Главные жалобы пользователей", "must": [], "expect_tools": ["top_issues"]},
  {"id": 15, "mode": "multi", "query": "Сравни рейтинги по платформам и назови топ-3 проблемы", "must": ["проблем"], "expect_tools": ["stats_by_platform", "top_issues"]},
  {"id": 16, "mode": "multi", "query": "Что хуже: дизайн или поддержка? Опирайся на отзывы и статистику", "must": [], "expect_tools": ["search_reviews"]},
  {"id": 17, "mode": "multi", "query": "Обзор VoC: платформы + главные pain points", "must": [], "expect_tools": []},
  {"id": 18, "mode": "pipeline", "query": "metrics", "must": [], "check": "ghost_rate"},
]


def _must_have(text: str, keys: list[str]) -> bool:
    if not keys:
        return True
    t = text.lower()
    return all(k.lower() in t for k in keys)


def _hit_sources(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    found = sum(1 for g in gold if any(g in r for r in retrieved))
    return found / len(gold)


def _path_cost(path: dict) -> float:
    pin = path.get("prompt_tokens", 0) or 0
    pout = path.get("completion_tokens", 0) or 0
    if pin == 0 and pout == 0:
        return path.get("cost_usd", 0.0) or 0.0
    return round(pin * 0.27 / 1e6 + pout * 1.10 / 1e6, 6)


def _is_api_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(x in s for x in ("401", "402", "403", "429", "insufficient balance", "unauthorized", "invalid api key"))


def _fail_api(msg: str) -> None:
    print(f"\nОШИБКА API: {msg}")
    if "402" in msg or "insufficient balance" in msg.lower():
        print("Пополните баланс: https://platform.deepseek.com")
    elif "401" in msg or "unauthorized" in msg.lower():
        print("Проверьте LLM_AUTH_TOKEN в .env")
    sys.exit(1)


def _ensure_api() -> None:
    ok, msg = check_api()
    if not ok:
        _fail_api(msg)


def run_case(case: dict) -> dict:
    if case["mode"] == "pipeline":
        m = json.loads((BASE / "output" / "metrics.json").read_text(encoding="utf-8"))
        ok = m.get("ghost_quote_rate", 1) <= 0.15 and m.get("valid_reviews", 0) >= 35
        path = {
            "steps": 0,
            "tools": ["pipeline"],
            "prompt_tokens": m.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": m.get("usage", {}).get("completion_tokens", 0),
            "cost_usd": m.get("cost_usd", 0),
        }
        return {"id": case["id"], "ok": ok, "path": path, "detail": m, "trace": []}

    if case["mode"] == "rag":
        ans, path = rag_ask(case["query"])
        hits = hybrid_retrieve(case["query"])["ids"]
        ok = _must_have(ans.answer, case["must"]) and _hit_sources(hits, case.get("gold_sources", [])) >= 0.5
        path["cost_usd"] = _path_cost(path)
        trace = [
            {"step": 1, "call": "hybrid_retrieve", "retrieved": path.get("retrieved", [])},
            {"step": 2, "final": ans.answer},
        ]
        return {
            "id": case["id"],
            "ok": ok,
            "answer": ans.answer[:200],
            "path": path,
            "hit_rate": _hit_sources(hits, case.get("gold_sources", [])),
            "trace": trace,
        }

    if case["mode"] == "agent":
        r = run_agent(case["query"], verbose=False)
        tools = r.get("tools", [])
        ok = bool(r.get("answer")) and _must_have(r.get("answer", ""), case["must"])
        if case.get("expect_tools"):
            ok = ok and all(t in tools for t in case["expect_tools"])
        usage = r.get("usage", {})
        path = {
            "steps": r.get("steps"),
            "tools": tools,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost_usd": r.get("cost_usd", _path_cost(usage)),
        }
        return {
            "id": case["id"],
            "ok": ok,
            "answer": (r.get("answer") or "")[:200],
            "path": path,
            "trace": r.get("trace", []),
        }

    r = run_multi(case["query"], verbose=False)
    tools = r.get("path", {}).get("tools", [])
    ok = bool(r.get("answer")) and _must_have(r.get("answer", ""), case["must"])
    if case.get("expect_tools"):
        ok = ok and all(t in tools for t in case["expect_tools"])
    path = dict(r.get("path", {}))
    if "cost_usd" not in path:
        path["cost_usd"] = _path_cost(path)
    return {
        "id": case["id"],
        "ok": ok,
        "answer": (r.get("answer") or "")[:200],
        "path": path,
        "trace": r.get("traces", []),
    }


def _write_traces(records: list[dict], cases: list[dict]) -> None:
    TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_OUT.open("w", encoding="utf-8") as f:
        for rec, case in zip(records, cases):
            f.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "mode": case["mode"],
                        "query": case["query"],
                        "ok": rec.get("ok"),
                        "path": rec.get("path") or {},
                        "trace": rec.get("trace", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="только RAG id 1-5")
    args = ap.parse_args()

    if not BM25_CACHE.exists():
        print("-> RAG ingest...")
        ingest()

    _ensure_api()

    cases = CASES if not args.quick else CASES[:5]
    results = []
    for c in cases:
        print(f"[{c['id']}] {c['mode']}: {c['query'][:50]}...")
        try:
            r = run_case(c)
        except Exception as e:
            if _is_api_error(e):
                _fail_api(str(e))
            raise
        results.append(r)
        print(f"   {'OK' if r.get('ok') else 'FAIL'}")

    passed = sum(1 for r in results if r.get("ok"))
    print(f"\nИтого: {passed}/{len(results)}")
    out = {"passed": passed, "total": len(results), "results": results}
    RESULTS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_traces(results, cases)
    print(f"Сохранено: {RESULTS_OUT}")
    print(f"Trace: {TRACE_OUT}")


if __name__ == "__main__":
    main()
