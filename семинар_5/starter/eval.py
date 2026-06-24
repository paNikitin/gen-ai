"""
Eval: 10 вопросов для макро-агента (ДЗ семинар 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый тест — один инструмент.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова get_fx_rate с разными датами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Multi-hop: ставка + ИПЦ + арифметика.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад при текущей ключевой ставке (правило 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Формула 72 / ставка.",
    },
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "compare_periods: fx_USD, два периода.",
    },
    {
        "id": 6,
        "query": "Насколько изменилась инфляция г/г между мартом 2023 и мартом 2024?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "compare_periods: cpi.",
    },
    {
        "id": 7,
        "query": "Как изменилась ключевая ставка с февраля 2022 по сегодня?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Трудный: «сегодня» vs историческая дата — агент может перепутать period_b.",
    },
    {
        "id": 8,
        "query": "Что было выше в марте 2024: инфляция или безработица?",
        "expected_tools": ["get_inflation", "get_unemployment"],
        "must_have": [],
        "comment": "Трудный: две разные метрики в %, легко перепутать порядок сравнения.",
    },
    {
        "id": 9,
        "query": "Сколько юаней за один доллар по кросс-курсу ЦБ сегодня?",
        "expected_tools": ["get_fx_rate", "calculate"],
        "must_have": [],
        "comment": "Реальный вопрос: кросс-курс через рубль.",
    },
    {
        "id": 10,
        "query": "Стоит ли сейчас держать сбережения в рублях с учётом реальной доходности вклада?",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": [],
        "comment": "Реальный вопрос: ставка, инфляция, субъективный вывод.",
    },
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(f"\n  tools used : {used_tools}")
    print(
        f"  expected    : {case['expected_tools']}  -> {'OK' if tool_match else 'MISS'}"
    )
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  -> {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Eval макро-агента (10 вопросов)")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--id", type=int, help="Запустить один кейс по id")
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    cases = [c for c in CASES if a.id is None or c["id"] == a.id]
    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in cases]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(cases)} пройдено")
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(f"  {mark} Q{r['id']} ({r['steps']} шагов) — {r['query'][:60]}")

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
