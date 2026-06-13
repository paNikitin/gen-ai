# Семинар 4 — RAG по корпусу статей Habr

Домашнее задание: гибридный RAG (dense + BM25 + RRF), сравнение чанкинга naive vs recursive.

## Быстрый старт

```bash
cd семинар_4
pip install -r requirements.txt
cp .env.example .env   # при необходимости — ключ DeepSeek для ask

# Скачать корпус (если data/ пуст)
python scripts/fetch_habr_corpus.py

# Сравнение стратегий чанкинга
python eval.py compare

# Вопрос к RAG
python pipeline.py ingest --strategy recursive
python pipeline.py ask "Что такое RAG?"
```

Корпус: 12 открытых статей с Habr (~314K символов) в `data/`. Gold-разметка — `data/gold.json`.
