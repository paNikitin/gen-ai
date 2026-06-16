"""
Pydantic-модели для паттерна Планировщик-Исполнитель-Критик.

Дизайн:
- Plan содержит reasoning + список SubQuestion.
- Каждый SubQuestion знает, какие tools ему нужны и от каких других
  подвопросов он зависит (depends_on).
- Исполнитель возвращает WorkerAnswer с сырым трейсом — raw_trace нужен
  для логов и для домашки (Schema-Validator в домашке С6).
- Критик возвращает Verdict: ok/не ok, причина, какое действие
  предпринять (accept / replan / rework).

Эти типы — единый контракт между 4 модулями. Если какой-то агент
возвращает невалидный JSON — клиент повторит запрос (max_retries).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """Один узкий подвопрос, который может решить Исполнитель за 1-2 шага."""

    id: int = Field(..., description="Порядковый номер подвопроса, начинается с 1")
    question: str = Field(
        ...,
        description="Конкретный вопрос формата «какой курс USD на 2022-01-01?».",
    )
    expected_tools: list[str] = Field(
        ...,
        description=(
            "Разрешённые инструменты для этого подвопроса. "
            "Подмножество {get_fx_rate, get_key_rate, get_inflation, calculate}."
        ),
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description=(
            "id подвопросов, ответ на которые нужно знать ДО исполнения этого. "
            "Пусто — подвопрос можно исполнять сразу."
        ),
    )


class Plan(BaseModel):
    """План декомпозиции исходного вопроса."""

    reasoning: str = Field(
        ...,
        description="2-3 предложения: почему именно такая декомпозиция.",
    )
    subquestions: list[SubQuestion] = Field(
        ...,
        description="Последовательность подвопросов. Может быть пустой, если вопрос нерешаем.",
    )


class WorkerAnswer(BaseModel):
    """Ответ Исполнителя на один подвопрос."""

    subquestion_id: int
    question_snippet: str = Field(
        ..., description="Первые ~60 символов вопроса — для компактных логов."
    )
    answer: str = Field(..., description="Короткий фактический ответ (1-2 фразы).")
    used_tools: list[str] = Field(
        default_factory=list,
        description="Имена tools, которые Исполнитель действительно вызвал.",
    )
    raw_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Сырой trace из run_agent() для диагностики и домашки.",
    )


class Verdict(BaseModel):
    """Вердикт Критика по набору ответов."""

    ok: bool = Field(..., description="True, если всё согласовано и корректно.")
    reason: str = Field(
        ..., description="Короткое объяснение, что именно понравилось/не понравилось."
    )
    action: Literal["accept", "replan", "rework"] = Field(
        ...,
        description=(
            "accept  — всё хорошо, финализируем;\n"
            "rework  — конкретные подвопросы переделать (см. rework_ids);\n"
            "replan  — план в корне неверен, переделать декомпозицию."
        ),
    )
    rework_ids: list[int] = Field(
        default_factory=list,
        description="Какие подвопросы переделать (только при action='rework').",
    )
