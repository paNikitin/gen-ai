"""
Pydantic-схема ответа RAG.

TODO (Блок 3 семинара): заполните поля RAGAnswer.
Сейчас схема пустая — модель возвращает строку, и это как раз проблема.
"""

from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str = Field(description="Итоговый ответ на вопрос")
    quotes: list[str] = Field(
        min_length=1, max_length=5, description="Точные цитаты из ретрива (1-5 )"
    )
    confidence: float = Field(
        ge=0, le=1, description="Уверенность модели. Если < 0.5 — возвращаем 'не знаю'"
    )
    sources: list[str] = Field(
        description="ID-чанков, откуда взяли, например: 'tbank_egor__0'"
    )
