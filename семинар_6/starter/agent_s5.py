"""
Макро-агент: ReAct + Tool Use API.

Сам агент — это `while step < max_iter: llm.call → execute tools → loop`.

Запуск:
    python agent.py "Какая реальная ключевая ставка сейчас?"
    python agent.py "Сравни курс USD сегодня и 2 января 2022"

Параметры модели — через .env (см. ../.env.example):
    LLM_BASE_URL=... LLM_AUTH_TOKEN=... LLM_MODEL=...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_model, make_raw_client
from schemas import TOOL_SCHEMAS
from tools import calculate, get_fx_rate, get_inflation, get_key_rate

TOOLS_IMPL = {
    "get_fx_rate": get_fx_rate,
    "get_key_rate": get_key_rate,
    "get_inflation": get_inflation,
    "calculate": calculate,
}


SYSTEM_PROMPT = """\
Ты — макроэкономический аналитик, работающий с актуальными данными ЦБ РФ и Росстата.
У тебя есть четыре инструмента. ЧИСЛА НИКОГДА НЕ ПРИДУМЫВАЙ — всегда получай их через tool calls.

Инструменты:
- get_fx_rate: курс валюты к рублю на дату
- get_key_rate: ключевая ставка ЦБ на дату
- get_inflation: ИПЦ (% г/г) на конец месяца
- calculate: безопасный калькулятор для арифметики над полученными числами

Алгоритм:
1. Разложи вопрос на подвопросы: какие числа нужны, в какой последовательности.
2. Для каждого числа — вызов соответствующего инструмента.
3. Арифметику считай ТОЛЬКО через calculate. Не пиши "21 - 9.5 = 11.5" в голове.
4. Реальная ставка = номинальная ставка − инфляция г/г (оба в процентах годовых).
5. Когда данных достаточно, выдай финальный ответ обычным текстом БЕЗ tool_calls.
   Одна-две фразы, с числами и единицами. Если число из fallback_csv — оговорись, что
   ЦБ в моменте недоступен и данные из локального архива.

Формат даты для инструментов — всегда YYYY-MM-DD.
"""


def run_agent(
    user_query: str,
    *,
    max_iter: int = 8,
    verbose: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Прогон ReAct-цикла. Возвращает {"answer": str, "trace": [...], "steps": int}."""
    client = make_raw_client()
    model = get_model()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    trace: list[dict[str, Any]] = []

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if verbose:
            print(f"[step {step}] tool_calls={len(msg.tool_calls or [])}")

        if not msg.tool_calls:
            trace.append({"step": step, "final": msg.content})
            return {"answer": msg.content, "trace": trace, "steps": step}

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {}
                obs: Any = {"error": f"invalid JSON arguments: {e}"}
            else:
                fn = TOOLS_IMPL.get(name)
                if fn is None:
                    obs = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        obs = fn(**args)
                    except TypeError as e:
                        obs = {"error": f"bad args for {name}: {e}"}
                    except Exception as e:
                        obs = {"error": f"{type(e).__name__}: {e}"}

            trace.append({"step": step, "call": name, "args": args, "obs": obs})
            if verbose:
                preview = json.dumps(obs, ensure_ascii=False)[:160]
                print(f"    {name}({args}) → {preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False),
                }
            )

    return {
        "answer": None,
        "trace": trace,
        "steps": max_iter,
        "error": f"agent exceeded max_iter={max_iter}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_agent(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(шагов: {res['steps']})")

    if args.trace:
        args.trace.write_text(
            json.dumps({"query": q, **res}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


if __name__ == "__main__":
    main()
