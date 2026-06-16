"""
Eval мульти-агента: 6 вопросов × 3 конфигурации.

Конфигурации:
  1. single  — одиночный агент С5
  2. pwc     — PWC без валидатора
  3. pwc_val — PWC + validate_plan

Запуск:
    python eval_pwc.py --single       # 1 прогон, быстро
    python eval_pwc.py -n 3           # N=3 (минимум для ДЗ)
    python eval_pwc.py -n 5           # полный прогон
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc, validate_plan

CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": "Ошибка C: арифметика без calculate. PWC + calculate-подвопрос.",
        "expected_tools": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["доллар"],
        "needs_calculate": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": "Ошибка B: поиск последнего месяца ИПЦ.",
        "expected_tools": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "needs_calculate": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": "Ошибка D: Планировщик галлюцинирует get_cumulative_inflation.",
        "expected_tools": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "needs_calculate": True,
        "validator_helps": True,
    },
    {
        "id": "Q4",
        "query": (
            "Посчитай накопленную инфляцию за 2024 год через get_yearly_cpi "
            "или аналогичный инструмент."
        ),
        "comment": "ГАРАНТИРОВАННО чинится валидатором: выдуманный get_yearly_cpi.",
        "expected_tools": {"get_inflation", "calculate"},
        "must_have_keywords": [],
        "needs_calculate": False,
        "validator_helps": True,
        "pass_if_no_plan_hallucination": True,
    },
    {
        "id": "Q5",
        "query": (
            "Какие сегодня официальные курсы USD, EUR и CNY к рублю по данным ЦБ?"
        ),
        "comment": "Естественная параллельность: 3 независимых get_fx_rate.",
        "expected_tools": {"get_fx_rate"},
        "must_have_keywords": ["доллар", "евро"],
        "needs_calculate": False,
        "parallel_friendly": True,
    },
    {
        "id": "Q6",
        "query": "Сколько юаней за один доллар по кросс-курсу ЦБ сегодня?",
        "comment": "Реальный вопрос: кросс-курс через рубль.",
        "expected_tools": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["юан"],
        "needs_calculate": True,
    },
]

VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _collect_tools_single(result: dict) -> set[str]:
    return {e["call"] for e in result.get("trace", []) if "call" in e}


def _collect_tools_pwc(result: dict) -> tuple[set[str], set[str]]:
    used: set[str] = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    plan_tools: set[str] = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    return used, plan_tools


def _check_single(case: dict, result: dict) -> dict:
    used = _collect_tools_single(result)
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_bad = case.get("needs_calculate") and "calculate" not in used and bool(ans)
    ok = bool(ans) and not hallucinated and must and not arith_bad
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "answer_preview": (result.get("answer") or "")[:160],
    }


def _check_pwc(case: dict, result: dict, *, with_validator: bool) -> dict:
    used, plan_tools = _collect_tools_pwc(result)
    ans = (result.get("answer") or "").lower()
    halluc_w = used - VALID_TOOL_NAMES
    halluc_p = plan_tools - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"]) if case["must_have_keywords"] else True

    if case.get("pass_if_no_plan_hallucination"):
        # Q4: успех = валидатор не допустил выдуманный инструмент в плане
        ok = not halluc_p and not halluc_w
        if with_validator and result.get("plan"):
            plan_errs = validate_plan(result["plan"])
            ok = ok and not plan_errs
    else:
        ok = (
            bool(result.get("answer"))
            and not halluc_w
            and not halluc_p
            and must
        )
        if case.get("needs_calculate") and "calculate" not in used and bool(ans):
            ok = False

    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(halluc_w),
        "hallucinated_in_plan": sorted(halluc_p),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or result.get("error") or "")[:160],
    }


def run_case(case: dict, *, n: int = 3) -> dict:
    buckets = {
        "single": {"runs": [], "pass": 0},
        "pwc": {"runs": [], "pass": 0},
        "pwc_val": {"runs": [], "pass": 0},
    }

    for _ in range(n):
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": str(e), "trace": []}
        c1 = _check_single(case, r1)
        buckets["single"]["runs"].append(c1)
        buckets["single"]["pass"] += int(c1["ok"])

        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": str(e), "trace": [], "plan": None}
        c2 = _check_pwc(case, r2, with_validator=False)
        buckets["pwc"]["runs"].append(c2)
        buckets["pwc"]["pass"] += int(c2["ok"])

        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": str(e), "trace": [], "plan": None}
        c3 = _check_pwc(case, r3, with_validator=True)
        buckets["pwc_val"]["runs"].append(c3)
        buckets["pwc_val"]["pass"] += int(c3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        **buckets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="1 прогон на кейс")
    ap.add_argument("-n", type=int, default=3, help="Прогонов на кейс (default 3)")
    ap.add_argument("--id", type=str, help="Только один кейс, напр. Q3")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    cases = [c for c in CASES if not args.id or c["id"] == args.id]
    print(f"Eval С6: {len(cases)} кейсов x {n} прогонов x 3 конфигурации\n")

    results = []
    for case in cases:
        print(f"=== {case['id']}: {case['query'][:65]}...")
        r = run_case(case, n=n)
        results.append(r)
        print(
            f"   single {r['single']['pass']}/{n}  "
            f"pwc {r['pwc']['pass']}/{n}  "
            f"pwc+val {r['pwc_val']['pass']}/{n}"
        )
        for run in r["pwc"]["runs"][:1]:
            if run.get("hallucinated_in_plan"):
                print(f"   ! pwc план: {run['hallucinated_in_plan']}")
        print()

    print("=" * 60)
    print("ИТОГО (доля pass):")
    for mode in ("single", "pwc", "pwc_val"):
        total = sum(r[mode]["pass"] for r in results)
        denom = len(results) * n
        print(f"  {mode:10s}: {total}/{denom}")

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
