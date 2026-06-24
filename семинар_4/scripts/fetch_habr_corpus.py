"""
Скачать статьи с Habr в data/*.txt
Запуск: python scripts/fetch_habr_corpus.py
"""
from __future__ import annotations

import re
import time
from html import unescape
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"

ARTICLES = [
    ("habr_931396_rag_obzor", "https://habr.com/ru/articles/931396/"),
    ("habr_1002152_rag_basics", "https://habr.com/ru/articles/1002152/"),
    ("habr_841428_chto_takoe_rag", "https://habr.com/ru/articles/841428/"),
    ("habr_862870_rag_langchain", "https://habr.com/ru/articles/862870/"),
    ("habr_988920_prompt_engineering", "https://habr.com/ru/articles/988920/"),
    ("habr_961088_vector_db_obzor", "https://habr.com/ru/articles/961088/"),
    ("habr_945404_tickets_rag", "https://habr.com/ru/articles/945404/"),
    ("habr_1000424_rag_production", "https://habr.com/ru/articles/1000424/"),
    ("habr_768844_llm_bez_haypa", "https://habr.com/ru/articles/768844/"),
    ("habr_897830_chromadb_deepseek", "https://habr.com/ru/companies/amvera/articles/897830/"),
    ("habr_929332_llm_obzor", "https://habr.com/ru/articles/929332/"),
    ("habr_776478_llm_agenty", "https://habr.com/ru/companies/ods/articles/776478/"),
]

MIN_CHARS = 3000


def extract_body(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = unescape(title_tag.get_text().split("/")[0].strip()) if title_tag else "Без названия"

    body_div = soup.find("div", class_=lambda c: c and "article-formatted-body" in c)
    if body_div is None:
        body_div = soup.find("div", id="publication-contents")
    if body_div is None:
        body_div = soup.find("article")
    if body_div is None:
        raise ValueError("Не найден блок статьи")

    for tag in body_div.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = body_div.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; gen-ai-hw/1.0)"}
    total = 0
    ok = 0

    with httpx.Client(headers=headers, follow_redirects=True, timeout=60) as client:
        for stem, url in ARTICLES:
            print(f"fetch {stem} ...", flush=True)
            resp = client.get(url)
            resp.raise_for_status()
            title, body = extract_body(resp.text)
            content = f"Источник: {url}\nЗаголовок: {title}\n\n{body}"
            out = DATA_DIR / f"{stem}.txt"
            out.write_text(content, encoding="utf-8")
            n = len(content)
            total += n
            ok += 1
            flag = "OK" if n >= MIN_CHARS else "SHORT"
            print(f"   {n} chars [{flag}]")
            time.sleep(1.2)

    print(f"\nDone: {ok} articles, {total} chars total")


if __name__ == "__main__":
    main()
