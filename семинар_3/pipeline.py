"""
Пайплайн анализа отзывов: IE → аспекты → Map-Reduce → LLM-as-judge.

Запуск:
    python pipeline.py input/
    python pipeline.py input/reviews.txt
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pydantic import ValidationError

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    ChunkSummary,
    JudgeReport,
    Review,
    ReviewSentiment,
    ReviewsSummary,
)

ALL_ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def _model() -> str:
    return get_model()

# Цены DeepSeek (примерно, $/1M tokens) — для оценки стоимости
_PRICE_IN = 0.27
_PRICE_OUT = 1.10


def load_corpus(input_path: str) -> str:
    p = Path(input_path)
    if p.is_dir():
        parts = sorted(p.glob("*.txt"))
        if not parts:
            raise FileNotFoundError(f"В {p} нет .txt файлов")
        return "\n\n".join(f.read_text(encoding="utf-8") for f in parts)
    return p.read_text(encoding="utf-8")


def split_by_review(corpus: str) -> list[str]:
    """Разбить корпус на блоки === REVIEW N ===."""
    chunks = re.findall(
        r"=== REVIEW \d+ ===.*?(?=\n=== REVIEW \d+ ===|\Z)",
        corpus,
        re.DOTALL,
    )
    return [c.strip() for c in chunks if c.strip()]


def _call(model_cls, system: str, user: str) -> tuple:
    return _get_client().chat.completions.create(
        model=_model(),
        response_model=model_cls,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": u.prompt_tokens or 0,
        "completion_tokens": u.completion_tokens or 0,
        "total_tokens": u.total_tokens or 0,
    }


def extract_reviews(corpus: str) -> tuple[list[Review], dict, int]:
    """IE: извлечь структурированные отзывы."""
    raw_items, resp = _call(list[Review], IE_SYSTEM, corpus)
    usage = _usage(resp)
    valid, errors = [], 0
    for item in raw_items:
        try:
            valid.append(Review.model_validate(item.model_dump()))
        except ValidationError:
            errors += 1
    return valid, usage, errors


def extract_aspects(corpus: str) -> tuple[list[ReviewSentiment], dict]:
    result, resp = _call(list[ReviewSentiment], ASPECTS_SYSTEM, corpus)
    return result, _usage(resp)


def check_quotes(
    aspects: list[ReviewSentiment],
    corpus: str,
) -> list[tuple[str, str]]:
    t = corpus.lower()
    ghosts: list[tuple[str, str]] = []
    for r in aspects:
        for a in r.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((r.author, a.quote))
    return ghosts


def build_heatmap(aspects: list[ReviewSentiment], out_path: Path) -> None:
    names = [r.author for r in aspects]
    sent_map = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(names), len(ALL_ASPECTS)), np.nan)
    for i, r in enumerate(aspects):
        for a in r.aspects:
            if a.aspect in ALL_ASPECTS:
                matrix[i, ALL_ASPECTS.index(a.aspect)] = sent_map[a.sentiment]
    plt.figure(figsize=(9, max(4, len(names) * 0.35)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=ALL_ASPECTS,
        yticklabels=names,
        center=0,
        cmap="RdYlGn",
        cbar_kws={"label": "sentiment"},
    )
    plt.title("Аспектная тональность по отзывам")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_chunk(chunk: str) -> tuple[ChunkSummary, dict]:
    result, resp = _call(ChunkSummary, CHUNK_SYSTEM, chunk)
    return result, _usage(resp)


def reduce_summaries(
    summaries: list[ChunkSummary],
    reduce_prompt: str = REDUCE_SYSTEM,
) -> tuple[ReviewsSummary, dict]:
    joined = "\n\n".join(
        f"## {s.author} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    result, resp = _call(ReviewsSummary, reduce_prompt, joined)
    return result, _usage(resp)


def summarize_reviews(corpus: str, workers: int = 6) -> tuple[ReviewsSummary, dict]:
    chunks = split_by_review(corpus)
    if len(chunks) <= 1:
        chunks = [corpus[i : i + 1200] for i in range(0, len(corpus), 1000)] or [corpus]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    summaries: list[ChunkSummary | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            i = futures[fut]
            summary, u = fut.result()
            summaries[i] = summary
            for k in usage_total:
                usage_total[k] += u[k]
    summary, u = reduce_summaries([s for s in summaries if s])
    for k in usage_total:
        usage_total[k] += u[k]
    return summary, usage_total


def build_judge_packet(reviews: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации (оцениваем)"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Issues из отзывов")
    for r in reviews:
        for issue in r.get("issues", []):
            parts.append(
                f"  - [{r['author']}/{issue['category']}, sev={issue['severity']}] "
                f"«{issue['quote']}»"
            )
    return "\n".join(parts)


def run_judge(reviews: list[dict], summary: dict) -> tuple[JudgeReport, dict]:
    evidence = build_judge_packet(reviews, summary)
    result, resp = _call(JudgeReport, JUDGE_SYSTEM, evidence)
    return result, _usage(resp)


def estimate_cost(usages: list[dict]) -> float:
    pin = sum(u["prompt_tokens"] for u in usages)
    pout = sum(u["completion_tokens"] for u in usages)
    return pin * _PRICE_IN / 1_000_000 + pout * _PRICE_OUT / 1_000_000


def analyze(input_path: str, out_dir: str = "output") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    usages: list[dict] = []

    corpus = load_corpus(input_path)
    n_reviews_in = len(split_by_review(corpus))

    print(f"→ IE: извлечение отзывов ({n_reviews_in} блоков)...")
    reviews, u, val_errors = extract_reviews(corpus)
    usages.append(u)
    reviews_data = [r.model_dump(mode="json") for r in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"   {len(reviews)} валидных, {val_errors} ValidationError")

    print("→ Аспектный анализ...")
    aspects, u = extract_aspects(corpus)
    usages.append(u)
    ghosts = check_quotes(aspects, corpus)
    aspects_data = [a.model_dump() for a in aspects]
    (out / "aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_heatmap(aspects, out / "heatmap.png")
    total_quotes = sum(len(r.aspects) for r in aspects)
    print(f"   ghost-цитат: {len(ghosts)} / {total_quotes}")

    print("→ Map-Reduce...")
    summary, u = summarize_reviews(corpus)
    usages.append(u)
    summary_dict = json.loads(summary.model_dump_json())
    (out / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("→ LLM-as-judge...")
    report, u = run_judge(reviews_data, summary_dict)
    usages.append(u)

    if report.overall_score < 0.7:
        print(f"   score={report.overall_score:.2f} < 0.7 — повтор REDUCE...")
        strict_reduce = REDUCE_SYSTEM + (
            "\n\nВАЖНО: каждая рекомендация должна напрямую следовать из жалоб."
            " Не предлагай функции, о которых пользователи не писали."
            " Все текстовые поля — строго на русском."
        )
        chunks = split_by_review(corpus)
        chunk_summaries = []
        for c in chunks:
            cs, u2 = summarize_chunk(c)
            usages.append(u2)
            chunk_summaries.append(cs)
        summary, u3 = reduce_summaries(chunk_summaries, strict_reduce)
        usages.append(u3)
        summary_dict = json.loads(summary.model_dump_json())
        (out / "summary.json").write_text(
            json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report, u4 = run_judge(reviews_data, summary_dict)
        usages.append(u4)

    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    cost = estimate_cost(usages)
    metrics = {
        "input_reviews": n_reviews_in,
        "valid_reviews": len(reviews),
        "validation_errors": val_errors,
        "ghost_quotes": len(ghosts),
        "total_aspect_quotes": total_quotes,
        "ghost_quote_rate": len(ghosts) / total_quotes if total_quotes else 0,
        "overall_score": report.overall_score,
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(cost, 4),
        "usage": {
            "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
            "completion_tokens": sum(u["completion_tokens"] for u in usages),
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== ИТОГ ===")
    print(summary.headline)
    print(f"Judge score: {report.overall_score:.2f}")
    print(f"Ghost quotes: {len(ghosts)}/{total_quotes}")
    print(f"Время: {elapsed:.1f}с, ~${cost:.4f}")
    print(f"Артефакты: {out}/")
    return metrics


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py <input/|file.txt> [output/]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
