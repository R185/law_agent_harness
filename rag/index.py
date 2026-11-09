from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from rag import config
from rag.api import ApiClient
from rag.bm25 import BM25Index
from rag.chunker import Chunk, chunk_document
from rag.metadata import Document


class KBIndex:
    """Unified knowledge base index supporting BM25 sparse and precomputed/cached dense vectors."""

    def __init__(self):
        self.docs: list[Document] = []
        self.chunks: list[Chunk] = []
        self.bm25: BM25Index = BM25Index()
        self.embeddings: np.ndarray | None = None  # shape (N, dim), dtype float16/float32
        self.chunk_hashes: list[str] = []
        self.emb_model: str = config.EMB_MODEL

    def build_chunks_and_bm25(self, docs: Sequence[Document]) -> None:
        """Process documents into chunks and build BM25 index."""
        self.docs = list(docs)
        self.chunks = []
        for d in self.docs:
            self.chunks.extend(chunk_document(d))

        self.bm25.fit(self.chunks)
        self.chunk_hashes = [
            hashlib.sha1(c.full.encode("utf-8")).hexdigest() for c in self.chunks
        ]

    def save(self, index_dir: Path) -> None:
        """Persist index metadata, chunks and embeddings to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        # 1. chunks.jsonl
        chunks_file = index_dir / "chunks.jsonl"
        with chunks_file.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                row = {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "seq": c.seq,
                    "header": c.header,
                    "text": c.text,
                    "section": c.section,
                    "clause_ids": c.clause_ids,
                    "notes": c.notes,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 2. docs.json
        docs_file = index_dir / "docs.json"
        with docs_file.open("w", encoding="utf-8") as f:
            doc_rows = [
                {
                    "doc_id": d.doc_id,
                    "path": d.path,
                    "title": d.title,
                    "category": d.category,
                    "doc_type": d.doc_type,
                    "meta": d.meta,
                    "removed_lines": d.removed_lines,
                }
                for d in self.docs
            ]
            json.dump(doc_rows, f, ensure_ascii=False, indent=2)

        # 3. embeddings.npy
        if self.embeddings is not None and len(self.embeddings) > 0:
            np.save(index_dir / "embeddings.npy", self.embeddings.astype(np.float16))

        # 4. manifest.json
        manifest = {
            "emb_model": self.emb_model,
            "chunker_version": "1.0",
            "total_chunks": len(self.chunks),
            "chunk_hashes": self.chunk_hashes,
        }
        with (index_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def load_precomputed_embeddings(self, index_dir: Path) -> bool:
        """Attempt to load embeddings.npy if chunk hashes and model match manifest."""
        manifest_file = index_dir / "manifest.json"
        emb_file = index_dir / "embeddings.npy"

        if not manifest_file.is_file() or not emb_file.is_file():
            return False

        try:
            with manifest_file.open(encoding="utf-8") as f:
                manifest = json.load(f)
            cached_hashes = manifest.get("chunk_hashes", [])
            cached_model = manifest.get("emb_model", "")

            # If chunks match exactly
            if cached_hashes == self.chunk_hashes and cached_model == config.EMB_MODEL:
                arr = np.load(emb_file)
                self.embeddings = arr.astype(np.float32)
                return True
        except Exception:
            pass
        return False

    def ensure_embeddings(self, client: ApiClient | None, index_dir: Path | None = None) -> None:
        """Ensure dense vectors exist for all chunks via cache or API."""
        if index_dir and self.load_precomputed_embeddings(index_dir):
            return

        # If no client or embedding key, dense search will be gracefully disabled
        if not client or not config.EMBED_KEY:
            return

        # Compute missing embeddings via API in batches of 32
        batch_size = 32
        texts = [c.full[:6000] for c in self.chunks]
        all_vecs: list[list[float]] = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = client.embeddings(batch, model=config.EMB_MODEL)
                all_vecs.extend(vecs)
            self.embeddings = np.array(all_vecs, dtype=np.float32)
            # Normalize just in case
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.embeddings = self.embeddings / norms
        except Exception as exc:
            print(f"[INDEX] Warning: failed to compute embeddings ({exc}); using BM25 only", file=sys.stderr)
            self.embeddings = None

    def dense_search(self, query: str, client: ApiClient | None, top_k: int = 30) -> list[tuple[int, float]]:
        """Dense similarity search using dot product."""
        if self.embeddings is None or client is None or not config.EMBED_KEY:
            return []

        try:
            q_vec_list = client.embeddings([query], model=config.EMB_MODEL)
            if not q_vec_list:
                return []
            q_vec = np.array(q_vec_list[0], dtype=np.float32)
            # L2 normalized dot product = cosine similarity
            scores = self.embeddings @ q_vec
            top_indices = np.argsort(-scores)[:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0.0]
        except Exception:
            return []
