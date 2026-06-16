"""
Критик: независимый проверяющий, который смотрит на набор ответов
и решает, принимаем или переделываем.

На семинаре нужно:
- передавать в промпт ТОЛЬКО answer + used_tools, а НЕ raw_trace (TODO 1);
- ставить temperature=0.7, не 0.0 (TODO 2).

Антипаттерны:
1. Передаём raw_trace → Критик «подсматривает» рассуждения Исполнителя
   и начинает с ними соглашаться.
2. temperature=0 → Критик работает как зеркало планировщика, повторяет
   ту же логику и всё «принимает».
"""

from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan, Verdict, WorkerAnswer

CRITIC_PROMPT = """\
Ты — критик мульти-агентной системы. Твоя работа — убедиться, что ответы
отвечают на исходный вопрос, согласованы между собой и получены честно.

Исходный вопрос пользователя:
  «{question}»

План, по которому работала система:
{plan_text}

Ответы Исполнителей (финальные, без трейса):
{answers_text}

Проверь ПОШАГОВО:
1. Все ли числа получены через calculate? Если в финальном ответе
   есть производное число (разность, отношение, произведение), но в
   used_tools соответствующего подвопроса НЕТ «calculate» — это БРАК.
2. Согласованы ли числа между подвопросами, на которые ссылаются последующие?
3. Покрывает ли план ВЕСЬ исходный вопрос? Если часть осталась без
   ответа — это replan.
4. Нет ли ответов вида «(ошибка: ...)» — они автоматически БРАК.

Вердикт:
- ok=True, action=accept — если всё чисто.
- ok=False, action=rework, rework_ids=[X] — если конкретные подвопросы
  нужно переделать.
- ok=False, action=replan — если план в принципе не охватывает вопрос.
"""


def critic(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> Verdict:
    plan_lines = []
    for sq in plan.subquestions:
        tools = ",".join(sq.expected_tools) or "—"
        deps = f" depends_on={sq.depends_on}" if sq.depends_on else ""
        plan_lines.append(f"  {sq.id}. [{tools}]{deps}  «{sq.question}»")
    plan_text = "\n".join(plan_lines) or "  (пустой план)"

    # TODO (блок 3.1): собрать answers_text из answers.
    # ОБЯЗАТЕЛЬНО передаём ТОЛЬКО answer + used_tools, НЕ raw_trace!
    # Иначе Критик начнёт подсматривать и соглашаться с Исполнителем.
    # Формат строки: "  <id>. [tool1,tool2] <answer>"

    ans_lines = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        tools = ",".join(a.used_tools) or "—"
        ans_lines.append(f"  {sq.id}. [{tools}] {a.answer}")

    answers_text = "\n".join(ans_lines) or "(ответов нет)"

    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": CRITIC_PROMPT.format(
                    question=question, plan_text=plan_text, answers_text=answers_text
                ),
            }
        ],
        response_model=Verdict,
        # TODO (блок 3.2): поставь temperature=0.7 (не 0.0).
        temperature=0.7,
        max_retries=2,
    )
