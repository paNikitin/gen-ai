"""
Наивный RAG: ChromaDB + OpenAI, fixed-size chunking, только dense-поиск.

Команды:
    python pipeline.py ingest
    python pipeline.py ask "Кто жаловался на push-уведомления?"

TODO для семинара:
    Блок 3, Фикс 1 — заменить фиксированные чанки на рекурсивные по абзацам
    Блок 3, Фикс 2 — обернуть ответ в Pydantic RAGAnswer
    Блок 3, Фикс 3 — добавить BM25-гибрид через rank-bm25 и RRF
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client import get_model, make_client, make_raw_client
from rank_bm25 import BM25Okapi
from schema import RAGAnswer

# Блок 1 — наивный RAG: ответ модели идёт обычным текстом
# client = make_raw_client()
client = make_client()
MODEL = get_model()
chroma = chromadb.PersistentClient(path="./chroma_db")

print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)
collection = chroma.get_or_create_collection(
    name="focus_groups",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)

DATA_DIR = Path(__file__).parent / "data"
BM25_CACHE = Path(__file__).parent / "bm25_cache.json"


splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)


def tokenize_ru(text: str):
    "Нормализация текста: приведение к нижнему регистру"
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text(text: str):
    "Разбивка текста на кусочки рекурсивным сплиттером"
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


# фиксированный чанкинг по символам
def chunk_text_naive(text: str, chunk_size: int = 2000) -> list[str]:
    """
    Примитивная нарезка: рубим каждые N символов.
    Проблема: граница может попасть в середину фразы «я ругался на |
    скорость» — и на запрос «скорость» не найдётся чанк про недовольство.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# заполнение векторного хранилища: читаем data/, режем, кладём в ChromaDB
def ingest():
    # Чистим старую коллекцию перед переиндексацией
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks = []
    all_ids = []
    all_meta = []

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i})

        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(json.dumps(bm25_data, ensure_ascii=False))

    total = collection.count()
    print(
        f"\nИндексировано: Dense — {total} чанков из {len(list(DATA_DIR.glob('*.txt')))} файлов"
    )
    print(f"\nBM25 — {len(all_ids)} чанков кэшировано в {BM25_CACHE.name}")


def _load_bm25():
    data = json.loads(BM25_CACHE.read_text())
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


# Retrieve + generate
def retrieve(query: str, k: int = 5) -> dict:
    """Dense-поиск в ChromaDB."""
    return collection.query(query_texts=[query], n_results=k)


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60) -> dict:
    """Hybrid-поиск контекста."""

    # семантический поиск
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    # tf-idf поиск
    bm25, bm25_ids, bm25_texts = _load_bm25()
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)

    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[
        :top
    ]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    # reciprocal rank fusion для совмещения результатов выдачи двух методов поиска
    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    # top-k списка
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    # достаем тексты по id
    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос продакта по архиву фокус-групп. "
        "Опирайся ТОЛЬКО на контекст ниже. Если в контексте нет ответа — "
        "скажи об этом прямо. Перечисли имена участников.\n\n"
        "Правила:\n"
        "1. Опирайся ТОЛЬКО на контекст ниже. Не добавляй факты из общего знания.\n"
        "2. В `quotes` — 1-5 точных коротких цитат (НЕ пересказ), с именами.\n"
        "3. В `sources` — id блоков, откуда взяты цитаты (формат: 'tbank_egor__0').\n"
        "4. В `confidence` — честная оценка: 0.9+ ТОЛЬКО когда прямой ответ в контексте,"
        "0.5-0.8, если собран из несколььких кусков, < 0.5 — если контекст не отвечает на запрос.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str):
    # Эмбеддим запрос и ищем топ-5 в Chroma.
    print("Поиск по базе...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, k=15)
    found = hits["ids"][0]
    print(
        f"   нашёл {len(found)} чанков за {time.time() - t0:.1f}с: {', '.join(found)}",
        flush=True,
    )

    # Кладём найденное в промпт, спрашиваем модель.
    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"   ответ за {time.time() - t1:.1f}с", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp)
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py {ingest|ask} [вопрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
