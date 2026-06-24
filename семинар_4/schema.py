from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str = Field(description="Итоговый ответ на вопрос на русском")
    quotes: list[str] = Field(min_length=1, max_length=5, description="Точные цитаты из контекста")
    confidence: float = Field(ge=0, le=1, description="Уверенность 0-1")
    sources: list[str] = Field(description="ID чанков, например habr_931396_rag_obzor__0")
