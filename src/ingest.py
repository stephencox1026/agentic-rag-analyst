"""Chunk + embed documents into a local Chroma vector store.

Usage:
    python -m src.ingest data/
"""
import sys
import pathlib

import chromadb
from openai import OpenAI

from . import config


def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(client: OpenAI, texts):
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main(data_dir: str):
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    chroma = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        chroma.delete_collection(config.COLLECTION)
    except Exception:
        pass
    coll = chroma.create_collection(config.COLLECTION)

    paths = sorted(pathlib.Path(data_dir).glob("**/*.md")) + sorted(
        pathlib.Path(data_dir).glob("**/*.txt")
    )
    if not paths:
        print(f"No .md/.txt files found in {data_dir}")
        return

    docs, ids, metas = [], [], []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            docs.append(chunk)
            ids.append(f"{p.name}::{i}")
            metas.append({"source": p.name, "chunk": i})

    # Embed in batches to stay under request limits.
    for batch_start in range(0, len(docs), 64):
        batch = docs[batch_start:batch_start + 64]
        vectors = embed(client, batch)
        coll.add(
            documents=batch,
            embeddings=vectors,
            ids=ids[batch_start:batch_start + 64],
            metadatas=metas[batch_start:batch_start + 64],
        )

    print(f"Ingested {len(docs)} chunks from {len(paths)} files into '{config.COLLECTION}'.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/")
