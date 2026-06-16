"""
Блок 6 — Супервизор: динамический диспетчер вместо статичного плана.

Контраст с PWC. Планировщик строит ВЕСЬ план заранее, оркестратор его исполняет.
Супервизор решает следующий шаг ПО ХОДУ: видит исходный вопрос и уже собранные
ответы и каждый раунд выбирает — задать ещё один подвопрос Исполнителю или закончить.

Плюс: гибкость. Минус: менее предсказуем и труднее ограничивается — поэтому
max_steps обязателен. Исполнитель переиспользуется из PWC без изменений.

На семинаре нужно закрыть 2 TODO.

Запуск:
    python supervisor.py "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_model, make_client
from schemas_pwc import SubQuestion, WorkerAnswer
from worker import worker


class SupervisorAction(BaseModel):
    """Решение супервизора на одном шаге."""

    action: Literal["ask", "finish"] = Field(
        ..., description="ask — задать подвопрос Исполнителю; finish — выдать финал."
    )
    question: str = Field(
        default="", description="Подвопрос Исполнителю (при action='ask')."
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Инструменты для подвопроса: подмножество {get_fx_rate, get_key_rate, get_inflation, calculate}.",
    )
    answer: str = Field(
        default="", description="Финальный ответ (при action='finish')."
    )


SUPERVISOR_SYSTEM = """\
Ты — супервизор макроэкономического агента. Тебе дают исходный вопрос и список
уже полученных ответов на подвопросы. Реши ОДНО следующее действие.

Инструменты, которые есть у Исполнителя: get_fx_rate, get_key_rate, get_inflation, calculate.

Правила:
- Если не хватает какого-то числа — action="ask": ОДИН узкий подвопрос + expected_tools.
- Если всех чисел достаточно — action="finish" и заполни answer (число + единица).
- Не повторяй уже заданные подвопросы.
"""
# - Любую арифметику (разности, отношения, проценты) делает Исполнитель через calculate отдельным подвопросом — сам в уме не считай.


def supervisor_step(
    question: str, answers: dict[int, WorkerAnswer]
) -> SupervisorAction:
    """Один вызов LLM: какое действие предпринять дальше."""
    done = (
        "\n".join(
            f"  {a.subquestion_id}. «{a.question_snippet}» → {a.answer} (tools={a.used_tools})"
            for a in answers.values()
        )
        or "  (пока ничего)"
    )
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {
                "role": "user",
                "content": f"Исходный вопрос: «{question}»\n\nУже собрано:\n{done}",
            },
        ],
        response_model=SupervisorAction,
        temperature=0.0,
        max_retries=2,
    )


def run_supervisor(
    question: str, *, max_steps: int = 6, verbose: bool = True
) -> dict[str, Any]:
    """Динамический цикл: шаг за шагом спрашиваем Исполнителя, пока супервизор не закончит."""
    answers: dict[int, WorkerAnswer] = {}
    trace: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        act = supervisor_step(question, answers)
        trace.append({"step": step, "action": act.action, "question": act.question})

        # TODO (блок 6.2): обработай решение супервизора.
        #   act.action == "finish" → верни {"answer": act.answer, "answers": answers,
        #                                    "trace": trace, "steps": step}
        #   act.action == "ask"    → собери SubQuestion(id=step, question=act.question,
        #                            expected_tools=act.expected_tools or ["calculate"],
        #                            depends_on=list(answers))  ← все прошлые ответы,
        #                            вызови worker(sq, prev_answers=answers),
        #                            положи в answers[step] и продолжи цикл.
        # Сейчас заглушка — сразу выходим, ничего не делая:
        if act.action == "finish":
            if verbose:
                print(f"[{step}] finish: {act.answer}")
            return {
                "answer": act.answer,
                "answers": answers,
                "trace": trace,
                "steps": step,
            }
        if verbose:
            print(f"[{step}] ask: [{','.join(act.expected_tools)}] {act.question}")
        sq = SubQuestion(
            id=step,
            question=act.question,
            expected_tools=act.expected_tools,
            depends_on=list(answers),
        )
        ans = worker(sq, prev_answers=answers)
        answers[step] = ans
        trace.append(
            {
                "step": step,
                "kind": "worker",
                "answer": ans.answer,
                "used_tools": ans.used_tools,
            }
        )
        if verbose:
            print(f" -> {ans.answer}, tool={ans.used_tools}")

    return {
        "answer": None,
        "error": f"супервизор не закончил за {max_steps} шагов",
        "answers": answers,
        "trace": trace,
        "steps": max_steps,
    }


if __name__ == "__main__":
    q = (
        " ".join(sys.argv[1:])
        or "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
    )
    res = run_supervisor(q)
    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(шагов: {res['steps']})")
