"""
RAG-пайплайн для ДЗ семинара 4.
Гибридный поиск (dense + BM25 + RRF), два режима чанкинга.

Команды:
    python pipeline.py ingest --strategy naive
    python pipeline.py ingest --strategy recursive
    python pipeline.py ask "Что такое hit-rate@5?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
from schema import RAGAnswer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_CACHE = BASE_DIR / "bm25_cache.json"
STRATEGY_FILE = BASE_DIR / ".chunking_strategy"

_client = None


def _get_llm():
    global _client
    if _client is None:
        from llm_client import get_model, make_client
        _client = (make_client(), get_model())
    return _client

RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(
    name="genai_corpus",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)


def chunk_text_naive(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def chunk_text_recursive(text: str) -> list[str]:
    return [c.strip() for c in RECURSIVE_SPLITTER.split_text(text) if c.strip()]


def chunk_document(text: str, strategy: str) -> list[str]:
    if strategy == "naive":
        return chunk_text_naive(text)
    if strategy == "recursive":
        return chunk_text_recursive(text)
    raise ValueError(f"Неизвестная стратегия: {strategy}")


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def ingest(strategy: str = "recursive") -> int:
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Нет .txt в {DATA_DIR}")

    for f in txt_files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_document(text, strategy)
        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i})
        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)
    BM25_CACHE.write_text(
        json.dumps({"ids": all_ids, "tokens": [tokenize_ru(c) for c in all_chunks], "texts": all_chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    STRATEGY_FILE.write_text(strategy, encoding="utf-8")

    total = collection.count()
    print(f"\nСтратегия: {strategy}")
    print(f"Индексировано: {total} чанков из {len(txt_files)} файлов")
    return total


def _load_bm25():
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60) -> dict:
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25()
    scores = bm25.get_scores(tokenize_ru(query))
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по архиву статей с Habr о LLM и RAG. "
        "Опирайся ТОЛЬКО на контекст ниже.\n\n"
        "Правила:\n"
        "1. Не добавляй факты из общего знания.\n"
        "2. quotes — 1-5 точных цитат из контекста.\n"
        "3. sources — id блоков (формат doc_name__N).\n"
        "4. confidence: 0.9+ при прямом ответе, <0.5 если контекст не помогает.\n"
        "5. Ответ на русском.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str) -> RAGAnswer:
    print("Поиск...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, k=5)
    found = hits["ids"][0]
    print(f"  {len(found)} чанков за {time.time() - t0:.1f}с: {', '.join(found)}", flush=True)

    print("Генерация...", flush=True)
    client, model = _get_llm()
    resp: RAGAnswer = client.chat.completions.create(
        model=model,
        response_model=RAGAnswer,
        max_retries=3,
        messages=[{"role": "user", "content": build_prompt(query, hits)}],
        temperature=0.2,
    )

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp.model_dump_json(indent=2, ensure_ascii=False))
    return resp


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--strategy", choices=["naive", "recursive"], default="recursive")

    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.strategy)
    elif args.cmd == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
