"""
Мультиагент (упрощённый PWC): планировщик → исполнители-агенты → синтез.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agent import _estimate_cost, _usage, run_agent
from llm_client import get_model, make_client, make_raw_client
from prompts import PLANNER_SYSTEM
from schema import Plan

VALID_TOOLS = {"search_reviews", "stats_by_platform", "top_issues"}


def validate_plan(plan: Plan) -> list[str]:
    errs = []
    for sq in plan.subquestions:
        for t in sq.expected_tools:
            if t not in VALID_TOOLS:
                errs.append(f"подвопрос {sq.id}: неизвестный инструмент «{t}»")
    return errs


def planner(question: str, feedback: str | None = None) -> Plan:
    client = make_client()
    msgs = [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": question}]
    if feedback:
        msgs.append({"role": "user", "content": f"Замечание: {feedback}"})
    return client.chat.completions.create(model=get_model(), response_model=Plan, max_retries=2, messages=msgs)


def run_multi(question: str, *, parallel: bool = True, verbose: bool = False) -> dict:
    plan = planner(question)
    errs = validate_plan(plan)
    if errs:
        plan = planner(question, feedback=f"Инструменты не существуют: {errs}")

    if verbose:
        print(f"[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. {sq.question}")

    if not plan.subquestions:
        return {
            "answer": plan.reasoning,
            "plan": plan,
            "workers": [],
            "path": {"steps": 1, "tools": [], "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            "traces": [],
        }

    def work(sq):
        r = run_agent(sq.question, max_iter=5, verbose=False)
        return {
            "id": sq.id,
            "answer": r.get("answer"),
            "tools": r.get("tools", []),
            "steps": r.get("steps", 0),
            "trace": r.get("trace", []),
            "usage": r.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        }

    if parallel and len(plan.subquestions) > 1:
        with ThreadPoolExecutor(max_workers=4) as ex:
            workers = list(ex.map(work, plan.subquestions))
    else:
        workers = [work(sq) for sq in plan.subquestions]

    bullets = "\n".join(f"- [{w['id']}] {w['answer']}" for w in workers)
    client = make_raw_client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": "Собери финальный ответ из фактов workers. Русский, 2-3 предложения."},
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{bullets}"},
        ],
        temperature=0.0,
    )
    final = resp.choices[0].message.content or ""
    all_tools = sorted({t for w in workers for t in w.get("tools", [])})
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for w in workers:
        u = w.get("usage", {})
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
    synth = _usage(resp)
    usage["prompt_tokens"] += synth["prompt_tokens"]
    usage["completion_tokens"] += synth["completion_tokens"]
    return {
        "answer": final,
        "plan": plan,
        "workers": workers,
        "traces": [{"worker": w["id"], "trace": w.get("trace", [])} for w in workers],
        "path": {
            "steps": sum(w.get("steps", 0) for w in workers) + 2,
            "tools": all_tools,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cost_usd": round(_estimate_cost(usage), 6),
        },
    }
