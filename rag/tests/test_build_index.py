"""
Unit tests for rag.build_index's plain-text ingestion path (chunk_text /
load_documents_from_text_file / build_index_from_text_file) -- the "Production RAG Streaming
Mode" entry point (see docs/PRODUCTION_RAG.md). `chunk_text`/`load_documents_from_text_file` are
pure-Python and always run; `build_index_from_text_file`'s end-to-end test monkeypatches the
embedder the same way rag/tests/test_retriever.py does, so it needs real `faiss` but not a real
embedding model download.
"""

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np

from rag.build_index import build_index_from_text_file, chunk_text, load_documents_from_text_file
from rag.retriever import Retriever

_FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


class TestChunkText(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\n   "), [])

    def test_short_single_paragraph_is_one_chunk(self):
        text = "This is a short paragraph that fits well under the chunk size."
        self.assertEqual(chunk_text(text, chunk_size_chars=800), [text])

    def test_splits_on_paragraph_boundaries_first(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size_chars=800)
        self.assertEqual(chunks, ["First paragraph.", "Second paragraph.", "Third paragraph."])

    def test_blank_paragraphs_are_dropped(self):
        text = "First.\n\n\n\nSecond.\n\n   \n\nThird."
        chunks = chunk_text(text, chunk_size_chars=800)
        self.assertEqual(chunks, ["First.", "Second.", "Third."])

    def test_long_paragraph_is_sub_split_with_overlap(self):
        paragraph = "x" * 1000
        chunks = chunk_text(paragraph, chunk_size_chars=400, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 400)
        # Reconstructing without overlap should recover at least the full original length's
        # worth of characters (overlap means some 'x's are double-counted, never lost).
        self.assertGreaterEqual(sum(len(c) for c in chunks), len(paragraph))

    def test_sub_split_chunks_cover_the_whole_paragraph(self):
        # Non-repeating content (sequential 3-digit numbers) so each chunk's text is unique and
        # `str.find` below can't match the wrong (earlier, identical-looking) occurrence.
        paragraph = "".join(f"{i:03d}" for i in range(150))  # 450 chars, all-unique substrings
        chunks = chunk_text(paragraph, chunk_size_chars=200, overlap_chars=50)
        # Every character of the original paragraph must appear in at least one chunk's span.
        covered = bytearray(len(paragraph))
        for chunk in chunks:
            idx = paragraph.find(chunk)
            self.assertNotEqual(idx, -1, f"chunk not found in original text: {chunk!r}")
            for i in range(idx, idx + len(chunk)):
                covered[i] = 1
        self.assertTrue(all(covered), "some part of the original paragraph was not covered by any chunk")


class TestLoadDocumentsFromTextFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.tmp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_produces_one_document_per_chunk_with_stable_ids(self):
        path = self._write("text.txt", "Para one.\n\nPara two.\n\nPara three.")
        documents = load_documents_from_text_file(path, chunk_size_chars=800)
        self.assertEqual([d.text for d in documents], ["Para one.", "Para two.", "Para three."])
        self.assertEqual(
            [d.doc_id for d in documents],
            ["text.txt-chunk-0", "text.txt-chunk-1", "text.txt-chunk-2"],
        )

    def test_empty_file_produces_no_documents(self):
        path = self._write("empty.txt", "")
        self.assertEqual(load_documents_from_text_file(path), [])


@unittest.skipUnless(_FAISS_AVAILABLE, "faiss is not installed")
class TestBuildIndexFromTextFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.tmp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_empty_file_raises_value_error(self):
        path = self._write("empty.txt", "   ")
        with self.assertRaises(ValueError):
            build_index_from_text_file(path, os.path.join(self.tmp_dir, "out"))

    def test_end_to_end_with_fake_embeddings_is_retrievable(self):
        text_path = self._write(
            "text.txt",
            "Cancellations made more than 24 hours before pickup receive a full refund.\n\n"
            "A refundable security deposit of $300 is required for the premium drone.\n\n"
            "Drones may not be flown in winds exceeding 20mph.",
        )
        out_path = os.path.join(self.tmp_dir, "production_index")

        vectors_by_text = {
            "Cancellations made more than 24 hours before pickup receive a full refund.": _unit([1.0, 0.0, 0.0]),
            "A refundable security deposit of $300 is required for the premium drone.": _unit([0.0, 1.0, 0.0]),
            "Drones may not be flown in winds exceeding 20mph.": _unit([0.0, 0.0, 1.0]),
        }

        # build_index_from_text_file constructs its own Retriever/EmbeddingModel internally, so
        # there's no instance to monkeypatch ahead of time -- patch at the class level instead
        # (same effect as test_retriever.py's per-instance patch, just applied before construction
        # since we don't control construction here). __init__/__post_init__ are left untouched;
        # only the two methods that would otherwise call the real (network-downloaded) model are
        # replaced, and restored in `finally` regardless of test outcome.
        from rag import embeddings as embeddings_module

        original_encode_passages = embeddings_module.EmbeddingModel.encode_passages
        original_encode_query = embeddings_module.EmbeddingModel.encode_query

        def fake_encode_passages(self, texts):
            return np.stack([vectors_by_text[t] for t in texts])

        def fake_encode_query(self, query):
            if "cancel" in query.lower():
                return vectors_by_text["Cancellations made more than 24 hours before pickup receive a full refund."]
            if "deposit" in query.lower():
                return vectors_by_text["A refundable security deposit of $300 is required for the premium drone."]
            return vectors_by_text["Drones may not be flown in winds exceeding 20mph."]

        embeddings_module.EmbeddingModel.encode_passages = fake_encode_passages
        embeddings_module.EmbeddingModel.encode_query = fake_encode_query
        try:
            report = build_index_from_text_file(text_path, out_path, chunk_size_chars=800)
            self.assertEqual(report["documents_indexed"], 3)

            retriever = Retriever()
            retriever.load_index(out_path)
            result = retriever.retrieve_context("How much is the deposit?", top_k=1)
            self.assertIn("deposit", result["contexts"][0].lower())
        finally:
            embeddings_module.EmbeddingModel.encode_passages = original_encode_passages
            embeddings_module.EmbeddingModel.encode_query = original_encode_query


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


if __name__ == "__main__":
    unittest.main()
