"""
Единая точка входа финального проекта NovaPay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input" / "reviews.txt"


def ensure_input():
    BASE.joinpath("input").mkdir(exist_ok=True)
    if not INPUT.exists():
        sys.exit(
            f"Нет {INPUT}. Корпус должен лежать в input/reviews.txt "
            "(см. input/README.md)."
        )
    from scripts.build_corpus import verify

    if not verify():
        sys.exit("Корпус повреждён или неполный. Проверьте input/reviews.txt")


def run_all():
    ensure_input()
    from pipeline import analyze
    from personas import run_focus_group
    from rag import ingest

    print("=" * 60 + "\n1/4 Пайплайн VoC\n" + "=" * 60)
    analyze(BASE / "input", BASE / "output")

    print("\n" + "=" * 60 + "\n2/4 Синтетическая фокус-группа\n" + "=" * 60)
    run_focus_group()

    print("\n" + "=" * 60 + "\n3/4 RAG-индекс\n" + "=" * 60)
    n = ingest()
    print(f"   {n} чанков")

    print("\n" + "=" * 60 + "\n4/4 Eval (можно отдельно: python eval.py)\n" + "=" * 60)
    print("   python eval.py --quick   # быстрая проверка")
    print("   python eval.py         # полный eval 18 кейсов")
    print(f"\nАртефакты: {BASE / 'output'}/")


def main():
    ap = argparse.ArgumentParser(description="Финальный проект VoC NovaPay")
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "analyze", "personas", "rag", "eval", "ask"])
    ap.add_argument("question", nargs="*", help="для ask")
    ap.add_argument("--quick-eval", action="store_true")
    a = ap.parse_args()

    ensure_input()
    if a.cmd == "all":
        run_all()
    elif a.cmd == "analyze":
        from pipeline import analyze
        analyze(BASE / "input", BASE / "output")
    elif a.cmd == "personas":
        from personas import run_focus_group
        run_focus_group()
    elif a.cmd == "rag":
        from rag import ingest
        print(f"Индекс: {ingest()} чанков")
    elif a.cmd == "eval":
        import eval as ev
        sys.argv = ["eval.py"] + (["--quick"] if a.quick_eval else [])
        ev.main()
    elif a.cmd == "ask":
        from rag import ingest, ask
        if not (BASE / "output" / "bm25_cache.json").exists():
            ingest()
        q = " ".join(a.question) or "Главные жалобы на приложение"
        ans, path = ask(q)
        print(ans.model_dump_json(indent=2, ensure_ascii=False))
        print("path:", path)


if __name__ == "__main__":
    main()
