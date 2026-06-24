"""
Eval по gold.json: hit-rate@5 на уровне документа-источника.

Команды:
    python eval.py --strategy naive
    python eval.py --strategy recursive
    python eval.py compare          # обе стратегии + eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import collection, hybrid_retrieve, ingest

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run_eval(k: int = 5, verbose: bool = True) -> dict:
    if collection.count() == 0:
        raise RuntimeError("Коллекция пуста. Запусти: python pipeline.py ingest --strategy ...")

    gold = load_gold()
    total = 0.0
    results = []

    for item in gold:
        hits = hybrid_retrieve(item["question"], k=k)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]
        score = hit_rate(retrieved_ids, item["gold_sources"])
        total += score
        results.append({
            "id": item["id"],
            "type": item["type"],
            "score": score,
            "gold": item["gold_sources"],
            "retrieved_sources": retrieved_sources,
            "question": item["question"],
        })
        if verbose:
            mark = "OK" if score == 1.0 else ("~" if score > 0 else "X")
            print(f"  [{item['id']:2d}] {item['type']:12s} hit@{k}={score:.2f} {mark}  {item['question'][:60]}")

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО hit-rate@{k} = {mean:.2f}")
    return {"mean": mean, "k": k, "results": results}


def compare(k: int = 5) -> None:
    output = {}
    for strategy in ("naive", "recursive"):
        print(f"\n{'='*60}\nСтратегия: {strategy.upper()}\n{'='*60}")
        ingest(strategy)
        output[strategy] = run_eval(k=k, verbose=True)

    print(f"\n{'='*60}\nСРАВНЕНИЕ\n{'='*60}")
    print(f"  naive:     hit-rate@{k} = {output['naive']['mean']:.2f}")
    print(f"  recursive: hit-rate@{k} = {output['recursive']['mean']:.2f}")
    winner = "recursive" if output["recursive"]["mean"] >= output["naive"]["mean"] else "naive"
    print(f"  Лучше: {winner}")

    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run", choices=["run", "compare"])
    parser.add_argument("--strategy", choices=["naive", "recursive"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "compare":
        compare(k=args.k)
        return

    if args.strategy:
        ingest(args.strategy)
    elif collection.count() == 0:
        print("Коллекция пуста → ingest recursive")
        ingest("recursive")

    run_eval(k=args.k)


if __name__ == "__main__":
    main()
