"""Проверка объёма корпуса: python scripts/check_corpus.py"""
from pathlib import Path

data = Path(__file__).parent.parent / "data"
files = sorted(data.glob("*.txt"))
total = sum(len(f.read_text(encoding="utf-8")) for f in files)
print(f"Документов: {len(files)}, символов: {total}")
for f in files:
    print(f"  {f.name}: {len(f.read_text(encoding='utf-8'))}")
