"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors: list[str] = []
    by_id = {sq.id: sq for sq in plan.subquestions}

    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(
                    f"подвопрос {sq.id}: неизвестный инструмент «{tool}»"
                )
        for dep in sq.depends_on:
            if dep not in by_id:
                errors.append(
                    f"подвопрос {sq.id}: depends_on ссылается на несуществующий id {dep}"
                )
            if dep == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит от самого себя")

    # цикл в depends_on
    visiting: set[int] = set()
    visited: set[int] = set()

    def dfs(node_id: int, path: list[int]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            errors.append(f"цикл depends_on: {' -> '.join(map(str, path + [node_id]))}")
            return
        if node_id not in by_id:
            return
        visiting.add(node_id)
        for dep in by_id[node_id].depends_on:
            dfs(dep, path + [node_id])
        visiting.remove(node_id)
        visited.add(node_id)

    for sq in plan.subquestions:
        dfs(sq.id, [])

    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни: внутри уровня зависимостей нет."""
    if not subqs:
        return []

    by_id = {s.id: s for s in subqs}
    remaining = set(by_id)
    done: set[int] = set()
    levels: list[list[SubQuestion]] = []

    while remaining:
        level = [
            by_id[sid]
            for sid in sorted(remaining)
            if all(dep in done for dep in by_id[sid].depends_on)
        ]
        if not level:
            raise ValueError("Цикл или битые ссылки в depends_on")
        levels.append(level)
        for sq in level:
            done.add(sq.id)
            remaining.remove(sq.id)
    return levels


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня (параллельно, если их несколько)."""
    if not level:
        return {}

    if parallel and len(level) > 1:
        out: dict[int, WorkerAnswer] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(level))) as ex:
            futs = {ex.submit(worker, sq, prev_answers): sq for sq in level}
            for fut in futs:
                sq = futs[fut]
                out[sq.id] = fut.result()
        return out

    return {sq.id: worker(sq, prev_answers) for sq in level}


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    if not answers:
        return plan.reasoning or "Нет данных для ответа."

    bullets = "\n".join(
        f"- [{i}] {answers[i].answer}" for i in sorted(answers)
    )
    client = make_raw_client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Собери из фактов короткий ответ пользователю (1-3 предложения). "
                    "Числа не придумывай — только из фактов. Если данных недостаточно, скажи честно."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос: {question}\n\nФакты:\n{bullets}",
            },
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or " · ".join(
        a.answer for a in answers.values()
    )


def _plan_with_validation(
    question: str,
    *,
    use_validator: bool,
    feedback: str | None = None,
    verbose: bool = True,
) -> Plan:
    plan = planner(question, feedback=feedback)
    if not use_validator:
        return plan

    errors = validate_plan(plan)
    if errors:
        if verbose:
            print(f"  [validator] ошибки плана: {errors}")
        plan = planner(
            question,
            feedback=f"Инструменты не существуют: {errors}",
        )
        errors2 = validate_plan(plan)
        if errors2 and verbose:
            print(f"  [validator] после переплана: {errors2}")
    return plan


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    plan = _plan_with_validation(question, use_validator=use_validator, verbose=verbose)
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
            "plan_errors": validate_plan(plan) if use_validator else [],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    answers: dict[int, WorkerAnswer] = {}

    for iter_num in range(1, max_iter + 1):
        answers = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            level_answers = execute_level(
                level, answers, parallel=parallel
            )
            answers.update(level_answers)
            for sq in level:
                ans = answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                        "parallel": parallel and len(level) > 1,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] -> {ans.answer[:120]}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "OK" if verdict.ok else "FAIL"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }

        if verdict.action == "replan":
            plan = _plan_with_validation(
                question,
                use_validator=use_validator,
                feedback=verdict.reason,
                verbose=verbose,
            )
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "replan",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            continue

        if verdict.action == "rework":
            plan = _plan_with_validation(
                question,
                use_validator=use_validator,
                feedback=(
                    f"Переделать подвопросы {verdict.rework_ids}: {verdict.reason}"
                ),
                verbose=verbose,
            )
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "rework",
                    "rework_ids": verdict.rework_ids,
                    "reasoning": plan.reasoning,
                }
            )
            continue

        break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="без параллели workers")
    ap.add_argument("--trace", type=Path, default=None)
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')}, {res.get('elapsed_sec', '?')} с)")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
