"""
Замер ускорения параллельного исполнения workers.

Запуск:
    python benchmark_parallel.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import run_pwc

Q1 = "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
Q5 = (
    "Какие сегодня официальные курсы USD, EUR и CNY к рублю по данным ЦБ?"
)


def _bench(query: str, *, parallel: bool, repeats: int = 2) -> list[float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_pwc(query, max_iter=2, verbose=False, parallel=parallel, use_validator=True)
        times.append(time.perf_counter() - t0)
    return times


def main():
    repeats = 2
    out = {}

    for label, query in [("Q1", Q1), ("Q5", Q5)]:
        seq = _bench(query, parallel=False, repeats=repeats)
        par = _bench(query, parallel=True, repeats=repeats)
        seq_mean = statistics.mean(seq)
        par_mean = statistics.mean(par)
        speedup = seq_mean / par_mean if par_mean > 0 else 0
        out[label] = {
            "query": query,
            "sequential_sec": round(seq_mean, 2),
            "parallel_sec": round(par_mean, 2),
            "speedup": round(speedup, 2),
            "raw_seq": [round(x, 2) for x in seq],
            "raw_par": [round(x, 2) for x in par],
        }
        print(f"\n{label}: sequential {seq_mean:.1f}s | parallel {par_mean:.1f}s | x{speedup:.2f}")

    path = Path(__file__).parent / "benchmark_parallel.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
