from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from taza_rag.llm import embed_texts
from taza_rag.models import Chunk


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    """Local dense (numpy) + BM25 index. Swap later for Postgres/pgvector."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        # L2-normalize for cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = self.embeddings / norms
        corpus_tokens = [tokenize(c.index_text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    def dense_search(self, query: str, k: int = 40) -> list[tuple[int, float]]:
        q = np.array(embed_texts([query])[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self.embeddings @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def sparse_search(self, query: str, k: int = 40) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        scores = np.array(self.bm25.get_scores(tokens), dtype=np.float32)
        k = min(k, len(scores))
        if k == 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        meta = {"n_chunks": len(self.chunks), "dim": int(self.embeddings.shape[1])}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> HybridIndex:
        embeddings = np.load(directory / "embeddings.npy")
        chunks: list[Chunk] = []
        with (directory / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate(json.loads(line)))
        return cls(chunks, embeddings)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> HybridIndex:
        texts = [c.index_text for c in chunks]
        vectors = embed_texts(texts)
        return cls(chunks, np.array(vectors, dtype=np.float32))
