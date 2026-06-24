"""
Сценарии для накопления trace.jsonl с разными типами ошибок (ДЗ, п.4).

Запуск:
    python scripts/seed_error_traces.py

Часть ошибок воспроизводится детерминированно через _exec_one (без LLM).
Если задан LLM_AUTH_TOKEN — дополнительно гоняются «ломающие» запросы к агенту.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import TRACE_PATH, _append_trace, _exec_one, run_agent  # noqa: E402


def _mock_tc(name: str, arguments: str, tc_id: str = "tc_mock") -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def seed_deterministic_errors() -> None:
    run_id = str(uuid.uuid4())

    # 1. Битый JSON в аргументах
    tc, args, obs = _exec_one(_mock_tc("calculate", '{"expression": (1+2}'))
    _append_trace(run_id, {"step": 1, "call": tc.function.name, "args": args, "obs": obs})

    # 2. Галлюцинация инструмента
    tc, args, obs = _exec_one(_mock_tc("get_exchange_rate", '{"currency": "USD"}'))
    _append_trace(run_id, {"step": 1, "call": tc.function.name, "args": args, "obs": obs})

    # 3. Инструмент упал — неверные аргументы
    tc, args, obs = _exec_one(_mock_tc("get_inflation", '{"year": 2024}'))
    _append_trace(run_id, {"step": 1, "call": tc.function.name, "args": args, "obs": obs})

    # 4. Ошибка внутри calculate
    tc, args, obs = _exec_one(_mock_tc("calculate", '{"expression": "import os"}'))
    _append_trace(run_id, {"step": 2, "call": tc.function.name, "args": args, "obs": obs})

    # 5. compare_periods — нет данных
    tc, args, obs = _exec_one(
        _mock_tc(
            "compare_periods",
            '{"metric": "cpi", "period_a": "1990-01", "period_b": "1990-06"}',
        )
    )
    _append_trace(run_id, {"step": 2, "call": tc.function.name, "args": args, "obs": obs})

    print(f"Детерминированные ошибки записаны в {TRACE_PATH} (run_id={run_id[:8]}...)")


def seed_agent_errors() -> None:
    if not os.environ.get("LLM_AUTH_TOKEN") and not os.environ.get("OPENAI_API_KEY"):
        print("LLM не настроен — пропускаем сценарии с агентом.")
        return

    scenarios = [
        ("max_iter", "Сравни USD, EUR, CNY, ключевую ставку и инфляцию за 2020-2024", 1),
        (
            "ambiguous",
            "Как изменилась ставка в марте?",
            6,
        ),
    ]
    for tag, query, max_iter in scenarios:
        print(f"\n--- agent scenario: {tag} ---")
        try:
            run_agent(query, max_iter=max_iter, verbose=True)
        except Exception as e:
            rid = str(uuid.uuid4())
            _append_trace(rid, {"step": 0, "error": f"{type(e).__name__}: {e}", "query": query})
            print(f"  exception logged: {e}")


def main() -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_deterministic_errors()
    seed_agent_errors()
    lines = TRACE_PATH.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nВсего строк в trace.jsonl: {len(lines)}")
    errors = []
    for line in lines[-20:]:
        row = json.loads(line)
        obs = row.get("obs") or {}
        if isinstance(obs, dict) and "error" in obs:
            errors.append(row.get("call", row.get("error", "?")))
        elif "error" in row:
            errors.append(row["error"][:40])
    print(f"Типы ошибок в хвосте лога: {errors}")


if __name__ == "__main__":
    main()
