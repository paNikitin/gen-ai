"""Pydantic-схемы финального проекта VoC."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

IssueCategory = Literal["performance", "design", "support", "price", "ads", "reliability"]
ReviewAspect = Literal["performance", "design", "support", "price", "ads", "reliability"]
Platform = Literal["App Store", "Google Play", "RuStore"]


class Issue(BaseModel):
    category: IssueCategory
    severity: int = Field(ge=1, le=5)
    quote: str


class Review(BaseModel):
    author: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    platform: Platform
    review_date: Optional[date] = None
    text: str = Field(min_length=10)
    issues: list[Issue]
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def date_not_in_future(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v > date.today():
            raise ValueError("дата отзыва не может быть в будущем")
        return v


class AspectSentiment(BaseModel):
    aspect: ReviewAspect
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    author: str
    aspects: list[AspectSentiment]


class ChunkSummary(BaseModel):
    author: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


class RAGAnswer(BaseModel):
    answer: str
    quotes: list[str] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(min_length=1)


class PersonaOpinion(BaseModel):
    name: str
    segment: str
    age: int = Field(ge=18, le=90)
    opinion: str
    rating: int = Field(ge=1, le=5)


class FocusGroup(BaseModel):
    topic: str
    opinions: list[PersonaOpinion] = Field(min_length=3, max_length=8)


class SubQuestion(BaseModel):
    id: int
    question: str
    expected_tools: list[str]


class Plan(BaseModel):
    reasoning: str
    subquestions: list[SubQuestion] = Field(default_factory=list)


class VoCAgentAnswer(BaseModel):
    answer: str
    tools_used: list[str] = Field(default_factory=list)
    steps: int = Field(ge=1, le=20)
