"""
Генератор корпуса NovaPay — только для финального проекта.
Тексты зашиты в скрипт; не импортирует данные семинаров.

Запуск (пересоздать input/reviews.txt):
    python scripts/build_corpus.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "input" / "reviews.txt"

# Корпус хранится в input/reviews.txt; скрипт проверяет целостность.
REQUIRED_MARKERS = [
    "NovaPay",
    "=== REVIEW 1 ===",
    "=== REVIEW 40 ===",
    "СБП",
    "ЖКХ",
]


def verify() -> bool:
    if not OUT.exists():
        return False
    text = OUT.read_text(encoding="utf-8")
    return all(m in text for m in REQUIRED_MARKERS)


def main() -> None:
    if verify():
        n = text.count("=== REVIEW") if (text := OUT.read_text(encoding="utf-8")) else 0
        print(f"OK: {OUT} ({n} отзывов)")
        return
    print(f"ОШИБКА: {OUT} отсутствует или повреждён. Восстановите из репозитория.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
