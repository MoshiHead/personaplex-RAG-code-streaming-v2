"""
Unit tests for the parts of rag.ws_demo_client that don't require a live server, a GPU, or torch
(`build_query_params`). The rest of that module -- the actual websocket round trip -- can only be
validated by running it against the real RunPod pod; see docs/PRODUCTION_RAG.md.
"""

import unittest

from rag.ws_demo_client import build_query_params


class TestBuildQueryParams(unittest.TestCase):
    def test_includes_required_fields_only_by_default(self):
        params = build_query_params(voice_prompt="NATM1.pt", text_prompt="You are a teacher.")
        self.assertEqual(params, {"voice_prompt": "NATM1.pt", "text_prompt": "You are a teacher."})

    def test_omits_seed_when_none(self):
        params = build_query_params(voice_prompt="v.pt", text_prompt="p", seed=None)
        self.assertNotIn("seed", params)

    def test_includes_seed_as_string_when_given(self):
        params = build_query_params(voice_prompt="v.pt", text_prompt="p", seed=42424242)
        self.assertEqual(params["seed"], "42424242")

    def test_omits_rag_query_when_empty(self):
        params = build_query_params(voice_prompt="v.pt", text_prompt="p", rag_query="")
        self.assertNotIn("rag_query", params)

    def test_includes_rag_query_when_given(self):
        params = build_query_params(voice_prompt="v.pt", text_prompt="p", rag_query="What is the deposit?")
        self.assertEqual(params["rag_query"], "What is the deposit?")


if __name__ == "__main__":
    unittest.main()
