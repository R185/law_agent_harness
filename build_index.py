from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag import config
from rag.api import ApiClient
from rag.chunker import chunk_document
from rag.index import KBIndex
from rag.loader import load_raw_documents
from rag.metadata import build_doc_graph, build_document_metadata
from rag.sanitize import sanitize_text


def build_index(
    dump: bool = False,
    embed: bool = False,
    index_dir_path: Path | None = None,
    kb_path: Path | None = None,
) -> None:
    target_dir = index_dir_path or (Path(__file__).resolve().parent / "index")
    print(f"[BUILD] Загрузка документов базы знаний...", flush=True)

    raw_docs = load_raw_documents(source_path=kb_path)
    docs = []
    for rd in raw_docs:
        clean_text, removed = sanitize_text(rd.text, rd.path)
        doc = build_document_metadata(rd.path, clean_text, removed)
        docs.append(doc)

    print(f"[BUILD] Документов: {len(docs)}", flush=True)

    kb_index = KBIndex()
    kb_index.build_chunks_and_bm25(docs)
    chunks = kb_index.chunks

    print(f"[BUILD] Построено чанков: {len(chunks)}", flush=True)

    if dump:
        print("=" * 85)
        print(f"{'Документ':48} | {'Чанков':6} | {'Мин':5} | {'Средн':5} | {'Макс':5}")
        print("-" * 85)
        doc_chunks_map: dict[str, list[int]] = {}
        for c in chunks:
            doc_chunks_map.setdefault(c.doc_id, []).append(len(c.text))
        for d in docs:
            lens = doc_chunks_map.get(d.doc_id, [0])
            print(f"{d.path[:48]:48} | {len(lens):6d} | {min(lens):5d} | {int(sum(lens)/len(lens)):5d} | {max(lens):5d}")
        print("=" * 85)

    client = None
    if embed:
        if not config.API_BASE or not config.EMBED_KEY:
            print("[BUILD] Внимание: API_BASE или EMBEDDER_API_KEY не заданы, эмбеддинги не рассчитываются", file=sys.stderr)
        else:
            client = ApiClient(timeout=90.0)

    kb_index.ensure_embeddings(client, index_dir=target_dir)

    kb_index.save(target_dir)
    print(f"[BUILD] Индекс успешно сохранён в {target_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка офлайн индекса базы знаний")
    parser.add_argument("--dump", action="store_true", help="Напечатать статистику чанкинга")
    parser.add_argument("--embed", action="store_true", help="Рассчитать эмбеддинги через API")
    parser.add_argument("--kb", type=Path, default=None, help="Путь к zip-архиву или папке базы знаний")
    parser.add_argument("--out", type=Path, default=None, help="Путь к директории индекса")
    args = parser.parse_args()

    build_index(dump=args.dump, embed=args.embed, index_dir_path=args.out, kb_path=args.kb)
