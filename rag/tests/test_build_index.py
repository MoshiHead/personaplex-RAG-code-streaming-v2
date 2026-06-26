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

    def test_merges_consecutive_short_paragraphs_into_one_chunk(self):
        # The actual fix (docs/PRODUCTION_RAG.md Section 9): short paragraphs that fit together
        # under chunk_size_chars are packed into ONE chunk, not split one-per-paragraph. A
        # document written with liberal blank lines for visual structure (headers, short
        # "Field: value" blocks, etc.) no longer fragments into many tiny chunks that could get
        # silently truncated downstream by a fixed chunk-count cap.
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size_chars=800)
        self.assertEqual(chunks, ["First paragraph.\n\nSecond paragraph.\n\nThird paragraph."])

    def test_keeps_paragraph_boundaries_as_chunk_boundaries_once_the_budget_is_exceeded(self):
        # Same input as above, but with a budget too small for all three to fit together --
        # paragraph boundaries are still respected as the only place a (non-oversized) chunk can
        # split.
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size_chars=20)
        self.assertEqual(chunks, ["First paragraph.", "Second paragraph.", "Third paragraph."])

    def test_blank_paragraphs_are_dropped(self):
        text = "First.\n\n\n\nSecond.\n\n   \n\nThird."
        # chunk_size_chars=10 is bigger than any single paragraph here (max 7 chars) but smaller
        # than any two combined (>= 6+2+6=14), so each stays its own chunk without being treated
        # as "oversized" (which would trigger sub-splitting instead of plain separation).
        chunks = chunk_text(text, chunk_size_chars=10)
        self.assertEqual(chunks, ["First.", "Second.", "Third."])

    def test_many_short_blank_line_separated_blocks_merge_instead_of_fragmenting(self):
        # Regression test for the real bug: a document that uses blank lines liberally for visual
        # structure (a title, several short "Entity: ..."/"Field: value" blocks, and "----"
        # divider lines) used to fragment into one chunk per block, for what is conceptually 2
        # entries -- which then silently lost the second entry entirely once downstream retrieval
        # capped the number of usable chunks (see docs/PRODUCTION_RAG.md Section 9 for the
        # real-world version of this with 14 blocks / 4 entries). With merging, this collapses to
        # far fewer chunks and both entries' facts end up close enough together to survive any
        # reasonable cap.
        text = (
            "Country Facts\n\n"
            "Entity: Alpha\nHolder: Person One\n\n"
            "Question: Who holds Alpha?\nAnswer: Person One holds Alpha.\n\n"
            "----------\n\n"
            "Entity: Beta\nHolder: Person Two\n\n"
            "Question: Who holds Beta?\nAnswer: Person Two holds Beta.\n\n"
            "----------"
        )
        naive_paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
        self.assertEqual(naive_paragraph_count, 7)  # what the old one-paragraph-one-chunk logic produced

        chunks = chunk_text(text, chunk_size_chars=800)
        self.assertLess(len(chunks), naive_paragraph_count)
        combined = "\n".join(chunks)
        self.assertIn("Person One", combined)
        self.assertIn("Person Two", combined)  # the entry that used to get truncated away

    def test_long_paragraph_is_sub_split_with_overlap(self):
        paragraph = "x" * 1000
        chunks = chunk_text(paragraph, chunk_size_chars=400, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 400)
        # Reconstructing without overlap should recover at least the full original length's
        # worth of characters (overlap means some 'x's are double-counted, never lost).
        self.assertGreaterEqual(sum(len(c) for c in chunks), len(paragraph))

    def test_overlap_chars_larger_than_chunk_size_chars_does_not_hang_or_crash(self):
        # Regression test: a misconfigured overlap_chars >= chunk_size_chars used to make `start`
        # in the sub-split loop go negative and never reach len(paragraph), looping until
        # MemoryError. The fix clamps the per-iteration step to at least 1 char of progress.
        paragraph = "y" * 50
        chunks = chunk_text(paragraph, chunk_size_chars=5, overlap_chars=150)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 5 for c in chunks))

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
        # chunk_size_chars=15 is bigger than any single paragraph here (max 11 chars) but smaller
        # than any two combined, so each stays its own chunk -- isolating doc_id assignment from
        # the merge behavior tested separately in TestChunkText.
        path = self._write("text.txt", "Para one.\n\nPara two.\n\nPara three.")
        documents = load_documents_from_text_file(path, chunk_size_chars=15)
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
            # chunk_size_chars=80 keeps these three (each individually under 80 chars, but no two
            # combined fit under 80) as three separate chunks -- this test is about retrieval
            # distinguishing between facts via real FAISS, not about merge behavior (tested
            # separately in TestChunkText).
            report = build_index_from_text_file(text_path, out_path, chunk_size_chars=80)
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
