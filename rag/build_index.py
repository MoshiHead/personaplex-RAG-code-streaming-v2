"""
Ingestion CLI for Phase 3/4 ("document ingestion", "create index", "save index"). Supports two
knowledge sources:

  --kb        a structured knowledge-base JSON file (a list of `{"doc_id", "topic", "text"}`
              objects -- see `rag/data/aero_rentals_kb.json`).
  --text-file a plain free-form text file (e.g. `text.txt`), automatically split into
              retrieval-sized chunks by `chunk_text()` -- this is the ingestion path for
              "Production RAG Streaming Mode" (see docs/PRODUCTION_RAG.md), where a knowledge base
              doesn't need to be hand-authored as structured JSON first.

Either way, the result is a FAISS index + metadata sidecar written to disk via
`rag.retriever.Retriever`.

Usage:
    python -m rag.build_index \
        --kb rag/data/aero_rentals_kb.json \
        --out rag_indexes/aero_rentals \
        --embedding-model bge-small \
        --vector-db faiss

    python -m rag.build_index \
        --text-file rag/data/text.txt \
        --out rag_indexes/production \
        --embedding-model bge-small \
        --vector-db faiss
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .retriever import Document, Retriever


def load_documents(kb_path: str) -> list[Document]:
    with open(kb_path, encoding="utf-8") as f:
        raw = json.load(f)
    documents = []
    for entry in raw:
        metadata = {k: v for k, v in entry.items() if k not in ("doc_id", "text")}
        documents.append(Document(text=entry["text"], doc_id=entry["doc_id"], metadata=metadata))
    return documents


def build_index(kb_path: str, out_path: str, embedding_model: str = "bge-small", vector_db: str = "faiss") -> dict:
    """Builds and saves an index from `kb_path`. Returns a small report dict (also what the CLI
    prints), useful for notebook cells that want to assert on it (e.g. "index has N documents")."""
    documents = load_documents(kb_path)

    t0 = time.monotonic()
    retriever = Retriever(embedding_model=embedding_model, vector_db=vector_db)
    n_indexed = retriever.build_index_from_documents(documents)
    build_time_s = time.monotonic() - t0

    retriever.save_index(out_path)

    return {
        "kb_path": kb_path,
        "out_path": out_path,
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "documents_indexed": n_indexed,
        "build_time_s": build_time_s,
    }


def chunk_text(text: str, chunk_size_chars: int = 800, overlap_chars: int = 150) -> list[str]:
    """Splits free-form text into retrieval-sized chunks. Paragraph-aware and merging: splits on
    blank lines first, then **packs consecutive paragraphs together** into one chunk up to
    `chunk_size_chars`, only starting a new chunk once the next paragraph would overflow the
    budget. A single paragraph that alone exceeds `chunk_size_chars` is sub-split with a sliding
    window (`overlap_chars` of overlap, so a fact split exactly across a cut still has a good
    chance of being fully visible in at least one chunk). Blank/whitespace-only paragraphs are
    dropped. Returns an empty list for empty/whitespace-only input.

    Merging matters for documents that use blank lines liberally for visual structure (headers,
    short "Field: value" blocks, "----" dividers, etc.) rather than as genuine topic boundaries --
    a naive "one paragraph = one chunk" policy fragments one logical entry into many tiny chunks,
    which silently loses content downstream wherever retrieval is capped at a fixed number of
    chunks (e.g. `RAGSession`'s no-query fallback) if the entry's chunks don't all fit under that
    cap in document order. See docs/PRODUCTION_RAG.md Section 9 for the real bug this caused.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size_chars:
            # Oversized paragraph: flush whatever's buffered first, then sub-split this one alone.
            if buffer:
                chunks.append(buffer)
                buffer = ""
            # `step` (not `chunk_size_chars - overlap_chars` inline) guards against a
            # misconfigured `overlap_chars >= chunk_size_chars`, which would otherwise make
            # `start` go non-increasing (or negative) and loop forever.
            step = max(chunk_size_chars - overlap_chars, 1)
            start = 0
            while start < len(paragraph):
                end = start + chunk_size_chars
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start += step
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= chunk_size_chars:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


def load_documents_from_text_file(
    text_path: str, chunk_size_chars: int = 800, overlap_chars: int = 150
) -> list[Document]:
    """Reads a plain text file and chunks it (via `chunk_text`) into `Document`s suitable for
    `Retriever.build_index_from_documents` -- the ingestion path for a free-form `text.txt`
    knowledge base, as opposed to `load_documents`'s structured KB JSON. `doc_id`s are
    `<basename>-chunk-<i>`, stable across rebuilds as long as the file's paragraph structure
    doesn't change.
    """
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
    basename = os.path.basename(text_path)
    return [Document(text=chunk, doc_id=f"{basename}-chunk-{i}") for i, chunk in enumerate(chunks)]


def build_index_from_text_file(
    text_path: str,
    out_path: str,
    embedding_model: str = "bge-small",
    vector_db: str = "faiss",
    chunk_size_chars: int = 800,
    overlap_chars: int = 150,
) -> dict:
    """Same as `build_index`, but ingests a plain text file (chunked via `chunk_text`) instead of
    a structured KB JSON -- the entry point for "Production RAG Streaming Mode", see
    docs/PRODUCTION_RAG.md. Returns the same report shape as `build_index`, plus the chunking
    parameters used."""
    documents = load_documents_from_text_file(text_path, chunk_size_chars, overlap_chars)
    if not documents:
        raise ValueError(f"{text_path!r} produced no chunks -- is the file empty?")

    t0 = time.monotonic()
    retriever = Retriever(embedding_model=embedding_model, vector_db=vector_db)
    n_indexed = retriever.build_index_from_documents(documents)
    build_time_s = time.monotonic() - t0

    retriever.save_index(out_path)

    return {
        "text_path": text_path,
        "out_path": out_path,
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "chunk_size_chars": chunk_size_chars,
        "overlap_chars": overlap_chars,
        "documents_indexed": n_indexed,
        "build_time_s": build_time_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kb", help="Path to a structured knowledge-base JSON file.")
    source.add_argument("--text-file", help="Path to a plain text file, chunked automatically.")
    parser.add_argument("--out", required=True, help="Output path prefix for the saved index.")
    parser.add_argument("--embedding-model", default="bge-small")
    parser.add_argument("--vector-db", default="faiss")
    parser.add_argument("--chunk-size-chars", type=int, default=800, help="Only used with --text-file.")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Only used with --text-file.")
    args = parser.parse_args()

    if args.kb:
        report = build_index(args.kb, args.out, args.embedding_model, args.vector_db)
    else:
        report = build_index_from_text_file(
            args.text_file, args.out, args.embedding_model, args.vector_db,
            args.chunk_size_chars, args.overlap_chars,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
