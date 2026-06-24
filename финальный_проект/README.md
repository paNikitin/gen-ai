# Финальный проект: VoC Intelligence для «NovaPay»

**Трек B** — прикладной конвейер анализа отзывов мобильного банка **NovaPay**.

Корпус **оригинальный** для этого проекта (`input/reviews.txt`, 40 отзывов). Данные семинаров не используются.

## Быстрый старт

```bash
cd финальный_проект
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env       # LLM_AUTH_TOKEN
python scripts/build_corpus.py  # проверка корпуса
python run.py                # полный прогон
python eval.py               # eval 18 кейсов
```

## Структура

```
финальный_проект/
├── run.py
├── pipeline.py, rag.py, agent.py, orchestrator.py, personas.py
├── eval.py
├── input/reviews.txt      ← корпус NovaPay (40 отзывов)
├── scripts/build_corpus.py
└── output/                ← артефакты после run.py
```

## После смены корпуса

Удалите старый индекс и перезапустите:

```bash
rmdir /s /q chroma_db
python run.py
```

Отчёт: `отчёт.md`

После `python eval.py`:
- `output/eval_results.json` — метрики eval
- `output/eval_trace.jsonl` — пошаговый trace

Eval требует рабочий API (проверка в начале). При 401/402 процесс останавливается.
