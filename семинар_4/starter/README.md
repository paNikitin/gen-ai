# Семинар 4 — RAG по архиву фокус-групп

## Что здесь

- `data/` — **20 транскриптов** фокус-групп (Олимп, BrightWord, Тинькофф) + `gold.json` с 10 размеченными вопросами для eval
- `pipeline.py` — **наивный RAG**: ChromaDB + fixed-size chunking + только dense-поиск. Специально неоптимальный — чинить будем на семинаре
- `schema.py` — **пустой** `RAGAnswer`. Заполним в Блоке 3
- `eval.py` — hit-rate@5 по `gold.json`

## Полезное

- [ChromaDB docs](https://docs.trychroma.com/)
- [LangChain text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)


# v1 avg hit-rate@5 = 0.27

## добавили RecursiveCharacterTextSplitter(256char)
# v2 avg hit-rate@5 = 0.5

## добавили RecursiveCharacterTextSplitter(512char)
# v2 avg hit-rate@5 = 0.53

## добавили RecursiveCharacterTextSplitter(8192char)
# v2 avg hit-rate@5 = 0.27


## добавили гибридный поиск с RRF
# v2 avg hit-rate@5 = 0.67
