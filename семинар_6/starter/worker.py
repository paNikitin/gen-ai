"""
Исполнитель: отвечает на ОДИН подвопрос, используя агента из С5 как функцию.

На семинаре нужно:
- собрать prev_context из depends_on (TODO).

Ключевая идея: Исполнитель видит ТОЛЬКО свой подвопрос + ответы тех
подвопросов, от которых он зависит. Это структурно ограничивает модель
и уменьшает соблазн галлюцинировать.
"""

from __future__ import annotations

from agent_s5 import run_agent
from schemas_pwc import SubQuestion, WorkerAnswer

WORKER_TEMPLATE = """\
Ты отвечаешь на ОДИН узкий вопрос, используя только разрешённые инструменты.

Вопрос: {question}

Разрешённые tools: {tools}

Контекст предыдущих подвопросов:
{prev_context}

Выдай короткий фактический ответ: одно предложение с числом и единицей.
Если подвопрос требует арифметики — ОБЯЗАТЕЛЬНО зови calculate.
Если число из fallback_csv — это нормально, просто упомяни в ответе.
"""


def worker(sq: SubQuestion, prev_answers: dict[int, WorkerAnswer]) -> WorkerAnswer:
    """Исполнить один подвопрос с учётом зависимостей."""
    # TODO (блок 2): собрать prev_context. Логика:
    #   если sq.depends_on непусто — для каждого dep_id достать
    #     prev_answers[dep_id] и собрать в строку вида
    #     "  <id>. «<question_snippet>» → <answer>"
    #   если sq.depends_on пусто — prev_context = "(нет зависимостей)".
    #   если какого-то dep_id нет в prev_answers — упомянуть «ответ недоступен».
    if sq.depends_on:
        lines = []
        for dep_id in sq.depends_on:
            if dep_id in prev_answers:
                a = prev_answers[dep_id]
                lines.append(f" {dep_id}.'{a.question_snippet}' -> {a.answer}")
            else:
                lines.append(f" {dep_id}.'(ответ недоступен)'")
        prev_context = "\n".join(lines)
    else:
        prev_context = "(нет зависимостей — независимый подвопрос)"

    prompt = WORKER_TEMPLATE.format(
        question=sq.question,
        tools=", ".join(sq.expected_tools) or "(ни одного — странно)",
        prev_context=prev_context,
    )

    result = run_agent(prompt, max_iter=5, verbose=False)
    used = [e["call"] for e in result.get("trace", []) if "call" in e]
    answer = result.get("answer") or f"(ошибка: {result.get('error', 'unknown')})"

    return WorkerAnswer(
        subquestion_id=sq.id,
        question_snippet=sq.question[:60],
        answer=answer,
        used_tools=used,
        raw_trace=result.get("trace", []),
    )


if __name__ == "__main__":
    sq = SubQuestion(
        id=1,
        question="Какой сейчас курс USD к рублю?",
        expected_tools=["get_fx_rate"],
    )
    ans = worker(sq, prev_answers={})
    print(f"answer: {ans.answer}")
    print(f"used_tools: {ans.used_tools}")
