# Production RAG Streaming Mode

What this is, why it's built the way it is, what was actually validated and how, and what you
need to do to point it at your own knowledge base. This is the productionization of the one
mechanism the Mode A-F research comparison (`docs/MODE_C_IMPLEMENTATION_REPORT.md`,
`docs/ARCHITECTURE_REPORT.md`) found to actually work, not a new injection mechanism.

## 1. What it is

A standing RAG setup for the live `moshi.server` (not just `moshi.offline`'s scripted runs):

1. A plain text file (`rag/data/text.txt` by default) is automatically chunked and embedded into a
   FAISS index -- no hand-authored structured KB JSON required.
2. The live server is started with `--rag-enable --rag-injection-mode persona_rag` pointed at that
   index.
3. Per connection, a `rag_query` parameter (already supported by `moshi.server` since the Mode C
   increment) triggers one retrieval + one `<system>`-wrapped injection burst, immediately after
   the persona/voice prompt and before any user audio is processed -- then the live duplex
   conversation proceeds completely normally, indistinguishable from a connection without RAG at
   all from that point on.

## 2. Why Mode C, and only Mode C

This is a deliberate constraint, not an oversight, backed by the full A-F comparison:

| Mode | Why it's excluded from production |
|---|---|
| A (baseline) | No retrieval at all -- the thing being productionized. |
| B (naive prompt template) | Confirmed negative control (Section 6): retrieves the same facts as C but doesn't engage with them at all. |
| D (turn injection) | Real-run result (Section 8): the burst itself doesn't leak, but the model abandons its in-progress sentence and re-greets instead of grounding. |
| E (dynamic/periodic injection) | Real-run result (Section 10): confirmed the `<system>`-tag hypothesis (no re-greet) but still doesn't ground -- the injected facts have no measurable effect once generation has already started. |
| F arm 2 (reset_and_replay) | Works, but costs ~1.5x arm 1's latency for no behavioral benefit over arm 1 -- there is no reason to ever choose this in production. |
| **C / F arm 1 (this mode)** | The only mechanism that reliably grounds, **and** it never resets the live RingKVCache. |

The cross-cutting finding from D and E (Section 10) is that injection *timing* relative to
generation -- before the model has sampled any part of its response, vs. after -- is what
actually determines whether injected knowledge gets used, not the `<system>`-tag format. That is
exactly what "once per connection, before generation starts" (Mode C's policy) guarantees and
what any mid-stream policy cannot.

## 3. Why "once per user turn" means "once per connection" here

PersonaPlex has no ASR anywhere in its pipeline (`docs/MODE_C_IMPLEMENTATION_REPORT.md` Section 2)
-- the only text ever available is the model's own sampled output, never a transcript of what the
user said. There is therefore no live query text to retrieve against mid-call. "Inject once per
user turn, never mid-stream" collapses to "inject once, at connection start, using the query
supplied via the `rag_query` connection parameter" -- which is exactly Mode C's existing, already
real-pod-validated design. Building genuine per-utterance retrieval would require bolting on a
separate ASR component listening to the same PCM stream, which is explicitly out of scope (no ASR
integration).

## 4. What was built

| File | Purpose |
|---|---|
| `rag/build_index.py` | Added `chunk_text()` (paragraph-aware chunking with overlap for long paragraphs), `load_documents_from_text_file()`, `build_index_from_text_file()`, and a `--text-file` CLI option alongside the existing `--kb`. No changes to `rag/retriever.py`/`rag/vector_store.py` -- they were already format-agnostic (`Document(text, doc_id, metadata)` in, FAISS index out), so plain-text ingestion only needed a new *front door*, not new retrieval machinery. |
| `rag/data/text.txt` | The default "automatically used" knowledge base -- the same AeroRentals facts already validated in Sections 3d/6, rewritten as flowing prose paragraphs instead of structured JSON, so there's a real paragraph structure for `chunk_text()` to split on. Replace this file's contents (or point `PRODUCTION_TEXT_KB_PATH` at a different file) to use your own knowledge base -- no code changes needed. |
| `rag/ws_demo_client.py` | A from-scratch Python WebSocket client for `moshi.server`'s `/api/chat` endpoint (mirrors the browser web UI's protocol exactly -- query params, handshake byte, Opus-encoded binary audio frames, text-token messages). Nothing in `rag/server_integration.py` changed -- the live server's RAG injection code path (`ServerState`/`handle_chat`, wired during the Mode C increment) already did everything this needs; this client just lets a notebook (or any Python script) drive it the same way a real user's browser would, instead of only being exercisable by hand via the web UI. |
| `PersonaPlex_RunPod_RTX5090.ipynb`, Section 22 | Builds the index from `text.txt`, launches a RAG-enabled live server, runs a real-time-paced demo query over an actual websocket connection, and prints the retrieved chunks / injection mechanism / final transcript / streaming latency as explicit proof of each success criterion. |

## 5. What was actually validated, and how

Consistent with this project's running discipline: this machine has no GPU, no CUDA, and none of
PersonaPlex's gated weights, so the real model cannot run here.

**Validated for real, right now:**
- `chunk_text`/`load_documents_from_text_file`/`build_index_from_text_file`: 10 unit tests
  (`rag/tests/test_build_index.py`), including an end-to-end ingest-then-retrieve round trip
  against the real `faiss` library (embedder monkeypatched, same pattern as
  `rag/tests/test_retriever.py` -- no network/model download needed).
- `rag/ws_demo_client.build_query_params`: 5 unit tests (`rag/tests/test_ws_demo_client.py`).
  `aiohttp`/`sphn`/`moshi` are all imported lazily inside functions specifically so this module
  (and these tests) never require any of them to be installed -- same discipline as
  `rag/embeddings.py`'s lazy `sentence_transformers` import.
- All 108 tests in `rag/tests/` pass; `moshi/moshi/offline.py`/`server.py` are unaffected (no
  changes to either file -- this feature only adds new files plus a `rag/build_index.py`
  extension).

**NOT validated here -- requires the real RunPod RTX 5090 pod (your next step):**
The actual claim this feature hinges on -- "a real websocket connection, driven by a plain
`text.txt` file, grounds its answer without interrupting streaming or resetting the connection" --
can only be checked by running Section 22 against the real server and model. The websocket
protocol implementation in `rag/ws_demo_client.py` was written by careful, exact mirroring of
`moshi/moshi/server.py`'s `handle_chat`/`opus_loop` (same message-kind bytes, same Opus codec
calls, same query parameters), not by testing it against a real server -- that mirroring could
still be wrong in a way only a real connection attempt would reveal (e.g. an Opus framing detail,
a timing assumption). Treat the first real run as the actual test of this module, not just of the
underlying (already-proven) injection mechanism.

## 6. Using your own knowledge base

Replace the contents of `rag/data/text.txt` with your own plain text (any paragraph structure
works -- `chunk_text` splits on blank lines first, then **packs consecutive short paragraphs
together** up to `chunk_size_chars` (default 800), only starting a new chunk once the next
paragraph would overflow the budget, and only sub-splits a single paragraph that alone exceeds
the budget), then re-run Section 22's "Build the production FAISS index" cell. To use a different
file path entirely, change `PRODUCTION_TEXT_KB_PATH` in that cell. No other code changes are
required -- retrieval, injection, and the live server's RAG code path are all already
knowledge-source-agnostic.

## 7. Performance expectations

Per Mode C's own real-pod benchmark (`docs/MODE_C_IMPLEMENTATION_REPORT.md` Section 3d/6):
retrieval + injection together cost roughly the 8-9 second range quoted in the brief for a
~5-document, ~340-token injected block at `bge-small` embedding speed -- this is a one-time,
connection-start cost, not a per-turn or per-frame cost, since the mechanism never re-injects or
resets mid-call. Larger knowledge bases or a larger embedding model will retrieve more slowly;
larger `top_k` injects more tokens (~25ms/token, per the same benchmark). Once injection
completes, streaming proceeds at the same speed as a connection with RAG disabled -- nothing in
this mechanism touches the per-frame `opus_loop` cost.

## 8. Real-pod bug: RAG never engaged for actual browser conversations (found and fixed)

The first real-pod test of this feature was through the actual browser web UI (real microphone,
real voice), not the notebook's scripted `rag.ws_demo_client` demo. The model gave generic,
ungrounded answers -- the symptom looked like `text.txt` wasn't "properly processed", but the
index and retrieval pipeline were both fine.

**Root cause**: `moshi.server`'s `rag_query` connection parameter only exists because *this
project* added it (Section 5's Mode C increment) -- the browser web UI predates it and has no
field to send one. PersonaPlex has no ASR, so the browser UI has genuinely no text to put there
even if it tried. `handle_chat`'s old code guarded injection on `rag_query` being truthy
(`if self.rag_session is not None and rag_query:`), so for every real conversation through the
browser, that condition was always false and RAG silently never engaged -- only the scripted demo
(which explicitly sets `rag_query=AERO_QUESTION_TEXT`) ever exercised it. This had been latent
since the Mode C increment; Section 22's own demo cell never caught it because it always supplies
an explicit query.

**Fix** (three changes, all in already-existing code paths, no new mechanism):

1. `RAGSession._retrieve_for_injection` (`rag/server_integration.py`) now accepts a falsy `query`
   and falls back to `Retriever.retrieve_all()` -- injecting knowledge-base chunks regardless of
   relevance -- instead of skipping injection. New `FaissVectorStore.get_all()`/
   `Retriever.retrieve_all()` (`rag/vector_store.py`, `rag/retriever.py`) support this by reading
   back stored chunks directly, bypassing similarity search entirely (there is nothing to rank
   against without a query). (This originally capped at `top_k`; Section 9 found and fixed a real
   bug in that cap.)
2. `moshi/moshi/server.py`'s `handle_chat` no longer requires `rag_query` to be truthy before
   attempting injection. A new `RAGConfig.default_query` / `--rag-default-query` lets an operator
   configure a real similarity-search fallback (e.g. a one-line description of the deployment's
   domain) for connections that don't supply their own query; when that's also empty, the
   whole-KB fallback above is what actually fires for a real browser connection.
3. `moshi/moshi/offline.py`'s `--rag-enable` no longer hard-requires `--rag-query` for
   `persona_rag`/`prompt_rag`/`cache_aware` (it still does for `turn_injection`/`dynamic_runtime`,
   whose "prepare" methods retrieve directly, without this fallback).

Section 22 of the notebook gained a verification cell ("Verify the fix: a connection with NO
query") that connects with `rag_query=""` -- exactly what the browser sends -- and asserts
injection still happened, instead of only ever testing the easy case.

**The remaining, genuine limitation**: this fix makes the model *always have* the knowledge base
in context, which is sufficient grounding for a knowledge base small enough to fit. It is **not**
true per-question retrieval -- there is still no live signal of what the user actually asked to
rank chunks against. For a knowledge base too large to inject in full, the practical levers are a
well-chosen `--rag-default-query` (static, but at least relevance-ranked) or, beyond the scope of
this project as currently constrained, adding ASR -- and even then, Sections 8/10's real-pod
findings suggest a turn-boundary-triggered mid-call injection would likely still arrive too late to
influence the response. There is currently no way around this without changing the "no ASR, no
mid-stream injection" constraints this project was built under.

## 9. Second real-pod bug: chunking fragmentation + a stale `top_k` cap silently dropped later entries

After Section 8's fix, a multi-entry knowledge base (Bangladesh political leadership: President,
Prime Minister, Speaker, Leader of the Opposition, four separate "Entity: .../Question: .../
Answer: ..." sections in `text.txt`) only ever grounded the **first** entry in the file, no matter
which entry was asked about. Reordering the file changed *which* entry worked, always the first
one -- a clean, deterministic, position-dependent symptom, not the vaguer "model sometimes
hallucinates" pattern Section 8's bug produced.

**Root cause, confirmed by direct reproduction against the real file**: two compounding bugs in
this project's own code, not a model limitation.

1. **Chunking was too fine-grained.** `chunk_text`'s original policy was "one paragraph (one
   blank-line-separated block) = one chunk." The Bangladesh file uses blank lines liberally for
   visual structure -- a title, four "Entity: ..." blocks, eight "Question:/Answer:" blocks, and
   three "----" dividers -- 14 blocks total for 4 conceptual entries. Run through the old
   `chunk_text`, the President's content alone (title + its Entity block + its two Q/A blocks +
   the first divider) occupied chunks 0-4.
2. **The no-query fallback was capped at `top_k` (default 5).** `RAGSession._retrieve_for_injection`
   called `Retriever.retrieve_all(limit=self.config.top_k)`, and `retrieve_all` returns chunks in
   plain file order. With 14 chunks and a cap of 5, only chunks 0-4 -- all about the title and the
   President -- were ever injected. The Prime Minister/Speaker/Opposition chunks (5-13) were never
   injected, period, regardless of which question was asked. Reordering the file just moved which
   entry landed in the surviving first 5 -- exactly the observed symptom.

Verified directly: running the real `rag/data/text.txt` through the old `chunk_text` produced
exactly 14 chunks, with chunks 0-4 covering only the title and President -- reproducing the bug
from the actual file content, not a synthetic approximation.

**Fix (two parts, addressing each root cause -- not a workaround for either):**

1. **`chunk_text` now merges consecutive short paragraphs** (`rag/build_index.py`) instead of
   giving every blank-line-separated block its own chunk: paragraphs are packed together up to
   `chunk_size_chars`, only starting a new chunk once the next paragraph would overflow the
   budget, and only sub-splitting a single paragraph that alone exceeds the budget. Run against
   the real `text.txt`, this collapses the same 14 blocks into 2 chunks. (A latent, unrelated bug
   in the sub-split loop -- `overlap_chars >= chunk_size_chars` made the scan position go
   non-increasing and loop until `MemoryError` -- was found and fixed in the same pass, via a
   `step = max(chunk_size_chars - overlap_chars, 1)` guard.)
2. **The no-query fallback is no longer capped by `top_k`.** New `RAGConfig.full_kb_max_chunks`
   (`int | None`, default `None`) is what `_retrieve_for_injection` passes to `retrieve_all()`
   now. `top_k` bounds a *ranked* similarity-search result, where cutting the lowest-ranked tail is
   reasonable; the no-query path has no ranking at all (chunks come back in plain file order), so
   reusing `top_k` as its cap was the wrong default -- it silently and deterministically drops
   whichever chunks happen to come later in the source document. The new default (`None`,
   uncapped) injects the entire knowledge base; set `--rag-full-kb-max-chunks` /
   `--rag-default-query` explicitly only once a knowledge base is large enough that injection
   latency (~25ms/token) becomes the actual constraint.

Either fix alone would have resolved this specific file (2 merged chunks comfortably fit under the
old `top_k=5` cap; an uncapped fallback would have injected all 14 fragments even unmerged) -- both
are implemented because each addresses a real, independent correctness gap: fragmentation produces
more chunks than necessary regardless of any cap, and capping the no-query path by a
similarity-search knob is conceptually wrong regardless of how well-chunked the source document is.

12 new/updated unit tests cover both fixes directly, including a reproduction of the
many-blank-line-block fragmentation pattern and the `overlap_chars >= chunk_size_chars` regression.
122 tests pass total.
