"""
RAG по отзывам: hybrid dense + BM25 + RRF.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from llm_client import get_model, make_client
from prompts import RAG_SYSTEM
from schema import RAGAnswer

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input"
CHROMA = BASE / "chroma_db"
BM25_CACHE = BASE / "output" / "bm25_cache.json"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", ". "])
EMBED = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
_chroma = chromadb.PersistentClient(path=str(CHROMA))
_collection = _chroma.get_or_create_collection("voc_reviews", embedding_function=EMBED, metadata={"hnsw:space": "cosine"})


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def _split_reviews(text: str) -> list[tuple[str, str]]:
    blocks = re.findall(r"(=== REVIEW \d+ ===.*?)(?=\n=== REVIEW \d+ ===|\Z)", text, re.DOTALL)
    out = []
    for b in blocks:
        b = b.strip()
        m = re.search(r"=== REVIEW (\d+) ===", b)
        rid = f"review_{m.group(1)}" if m else "review"
        out.append((rid, b))
    return out


def ingest() -> int:
    CHROMA.parent.mkdir(parents=True, exist_ok=True)
    (BASE / "output").mkdir(exist_ok=True)
    existing = _collection.get()
    if existing["ids"]:
        _collection.delete(ids=existing["ids"])

    chunks, ids, meta = [], [], []
    for f in sorted(INPUT.glob("*.txt")):
        for rid, block in _split_reviews(f.read_text(encoding="utf-8")):
            for i, c in enumerate(SPLITTER.split_text(block)):
                cid = f"{rid}__{i}"
                chunks.append(c)
                ids.append(cid)
                meta.append({"source": rid})

    _collection.add(documents=chunks, ids=ids, metadatas=meta)
    BM25_CACHE.write_text(
        json.dumps({"ids": ids, "tokens": [tokenize_ru(c) for c in chunks], "texts": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(chunks)


def _load_bm25():
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5) -> dict:
    dense = _collection.query(query_texts=[query], n_results=15)
    dense_ids = dense["ids"][0]
    bm25, bm25_ids, texts = _load_bm25()
    scores = bm25.get_scores(tokenize_ru(query))
    sparse_ids = [bm25_ids[i] for i in sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:15]]
    rrf: dict[str, float] = {}
    c = 60
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (c + rank)
    top = [x for x, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    text_by = dict(zip(bm25_ids, texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by[did] = dense["documents"][0][i]
    return {"ids": top, "documents": [text_by[i] for i in top]}


def ask(query: str) -> tuple[RAGAnswer, dict]:
    hits = hybrid_retrieve(query)
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(hits["ids"], hits["documents"]))
    client = make_client()
    resp = client.chat.completions.create(
        model=get_model(),
        response_model=RAGAnswer,
        max_retries=3,
        temperature=0.1,
        with_completion=True,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Контекст:\n{ctx}\n\nВопрос: {query}"},
        ],
    )
    answer, completion = resp
    u = getattr(completion, "usage", None)
    path = {"retrieved": hits["ids"], "steps": 2, "tools": ["hybrid_retrieve", "llm"]}
    if u:
        path["prompt_tokens"] = u.prompt_tokens
        path["completion_tokens"] = u.completion_tokens
    return answer, path
