"""
Пайплайн VoC: IE → аспекты → Map-Reduce → LLM-as-judge + ghost-цитаты.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pydantic import ValidationError

from llm_client import get_model, make_client
from prompts import ASPECTS_SYSTEM, CHUNK_SYSTEM, IE_SYSTEM, JUDGE_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, JudgeReport, Review, ReviewSentiment, ReviewsSummary

BASE = Path(__file__).resolve().parent
ALL_ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]
_PRICE_IN, _PRICE_OUT = 0.27, 1.10
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def _call(model_cls, system: str, user: str, **kw):
    return _get_client().chat.completions.create(
        model=get_model(),
        response_model=model_cls,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kw,
    )


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {"prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0}


def load_corpus(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        parts = sorted(p.glob("*.txt"))
        return "\n\n".join(f.read_text(encoding="utf-8") for f in parts)
    return p.read_text(encoding="utf-8")


def split_by_review(corpus: str) -> list[str]:
    chunks = re.findall(r"=== REVIEW \d+ ===.*?(?=\n=== REVIEW \d+ ===|\Z)", corpus, re.DOTALL)
    return [c.strip() for c in chunks if c.strip()]


def extract_reviews(corpus: str):
    items, resp = _call(list[Review], IE_SYSTEM, corpus)
    valid, errors = [], 0
    for item in items:
        try:
            valid.append(Review.model_validate(item.model_dump()))
        except ValidationError:
            errors += 1
    return valid, _usage(resp), errors


def extract_aspects(corpus: str):
    result, resp = _call(list[ReviewSentiment], ASPECTS_SYSTEM, corpus)
    return result, _usage(resp)


def check_quotes(aspects: list[ReviewSentiment], corpus: str) -> list[tuple[str, str]]:
    t = corpus.lower()
    ghosts = []
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
    sns.heatmap(matrix, annot=True, fmt=".0f", xticklabels=ALL_ASPECTS, yticklabels=names, center=0, cmap="RdYlGn")
    plt.title("Аспектная тональность")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_reviews(corpus: str, workers: int = 6):
    chunks = split_by_review(corpus) or [corpus]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    summaries: list[ChunkSummary | None] = [None] * len(chunks)

    def chunk_one(c):
        r, resp = _call(ChunkSummary, CHUNK_SYSTEM, c)
        return r, _usage(resp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(chunk_one, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            i = futs[fut]
            s, u = fut.result()
            summaries[i] = s
            usage["prompt_tokens"] += u["prompt_tokens"]
            usage["completion_tokens"] += u["completion_tokens"]

    joined = "\n\n".join(
        f"## {s.author} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries if s
    )
    summary, resp = _call(ReviewsSummary, REDUCE_SYSTEM, joined)
    u2 = _usage(resp)
    usage["prompt_tokens"] += u2["prompt_tokens"]
    usage["completion_tokens"] += u2["completion_tokens"]
    return summary, usage


def run_judge(reviews: list[dict], summary: dict):
    lines = ["## action_items"] + [f"  {i}. {a}" for i, a in enumerate(summary.get("action_items", []), 1)]
    lines.append("\n## issues")
    for r in reviews:
        for iss in r.get("issues", []):
            lines.append(f"  [{r['author']}/{iss['category']}] «{iss['quote']}»")
    result, resp = _call(JudgeReport, JUDGE_SYSTEM, "\n".join(lines))
    return result, _usage(resp)


def estimate_cost(usages: list[dict]) -> float:
    pin = sum(u["prompt_tokens"] for u in usages)
    pout = sum(u["completion_tokens"] for u in usages)
    return pin * _PRICE_IN / 1e6 + pout * _PRICE_OUT / 1e6


def analyze(input_path: str | Path = None, out_dir: str | Path = None) -> dict:
    input_path = input_path or BASE / "input"
    out = Path(out_dir or BASE / "output")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    usages = []

    corpus = load_corpus(input_path)
    n_in = len(split_by_review(corpus))
    print(f"-> IE ({n_in} отзывов)...")
    reviews, u, val_err = extract_reviews(corpus)
    usages.append(u)
    reviews_data = [r.model_dump(mode="json") for r in reviews]
    (out / "reviews.json").write_text(json.dumps(reviews_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-> Аспекты + ghost-check...")
    aspects, u = extract_aspects(corpus)
    usages.append(u)
    ghosts = check_quotes(aspects, corpus)
    (out / "aspects.json").write_text(
        json.dumps([a.model_dump() for a in aspects], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_heatmap(aspects, out / "heatmap.png")
    total_q = sum(len(r.aspects) for r in aspects)

    print("-> Map-Reduce...")
    summary, u = summarize_reviews(corpus)
    usages.append(u)
    summary_dict = json.loads(summary.model_dump_json())
    (out / "summary.json").write_text(json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-> LLM-as-judge...")
    report, u = run_judge(reviews_data, summary_dict)
    usages.append(u)
    (out / "judge_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    metrics = {
        "input_reviews": n_in,
        "valid_reviews": len(reviews),
        "validation_errors": val_err,
        "ghost_quotes": len(ghosts),
        "total_aspect_quotes": total_q,
        "ghost_quote_rate": round(len(ghosts) / total_q, 3) if total_q else 0,
        "overall_score": report.overall_score,
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(estimate_cost(usages), 4),
        "usage": {"prompt_tokens": sum(u["prompt_tokens"] for u in usages), "completion_tokens": sum(u["completion_tokens"] for u in usages)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "ghost_quotes.json").write_text(json.dumps([{"author": a, "quote": q} for a, q in ghosts], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: ghost {len(ghosts)}/{total_q}, judge={report.overall_score:.2f}, ${metrics['cost_usd']}")
    return metrics
