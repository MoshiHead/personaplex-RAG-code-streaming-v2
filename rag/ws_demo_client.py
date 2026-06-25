"""
Lightweight Python WebSocket client for demonstrating "Production RAG Streaming Mode" against the
*live* `moshi.server` -- as opposed to driving the same connection-start injection mechanism
through `moshi.offline`, which never touches the actual websocket protocol the real product
(browser web UI) uses. See docs/PRODUCTION_RAG.md.

No GPU or model weights are needed on the client side -- only `moshi.models.loaders.SAMPLE_RATE`/
`FRAME_RATE` (plain module-level constants, no weight download) and `sphn`'s Opus codec (the same
library `moshi.server` itself uses for the wire format), plus `aiohttp` for the websocket. All
three (`moshi`, `sphn`, `aiohttp`) are imported lazily, inside functions, so importing this module
itself -- e.g. to unit-test `build_query_params` -- never requires any of them to be installed,
the same discipline `rag/embeddings.py` uses for `sentence_transformers`. This mirrors
`moshi/moshi/server.py`'s `handle_chat` protocol exactly:

  - connect to ws(s)://<host>:<port>/api/chat?voice_prompt=...&text_prompt=...&seed=...&rag_query=...
  - first message from the server: a single handshake byte, b"\\x00"
  - client -> server audio: b"\\x01" + <opus-encoded bytes>
  - server -> client audio: b"\\x01" + <opus-encoded bytes>; text: b"\\x02" + <utf-8 text>

This module is the one piece of the production-RAG feature that genuinely cannot be unit-tested
without a live server + GPU + loaded model -- consistent with how `moshi/moshi/offline.py` and
`moshi/moshi/server.py` themselves are validated in this project (syntax-checked + manually
reviewed here; the actual claim -- "the live websocket connection grounds answers in `text.txt`
without interrupting streaming" -- can only be confirmed by running it against the real pod).
"""

from __future__ import annotations

import asyncio
import json
import time
import wave
from typing import Optional

import numpy as np


def _audio_constants() -> tuple[int, int]:
    """Returns (SAMPLE_RATE, FRAME_SIZE). Imported lazily, inside a function, so importing this
    module (e.g. to use `build_query_params` in a unit test) never requires `moshi`/`torch` to be
    installed -- the same discipline `rag/embeddings.py` uses for `sentence_transformers`."""
    from moshi.models.loaders import FRAME_RATE, SAMPLE_RATE

    return SAMPLE_RATE, int(SAMPLE_RATE / FRAME_RATE)


def build_query_params(
    voice_prompt: str, text_prompt: str, rag_query: str = "", seed: Optional[int] = None
) -> dict:
    """Builds the `/api/chat` query-string params exactly as `moshi.server.handle_chat` reads
    them (`request.query["voice_prompt"]`, `request.query["text_prompt"]`, `request["seed"]`,
    `request.query.get("rag_query", "")`). Factored out from `run_streaming_query` so it's testable
    without a live server.
    """
    params = {"voice_prompt": voice_prompt, "text_prompt": text_prompt}
    if seed is not None:
        params["seed"] = str(seed)
    if rag_query:
        params["rag_query"] = rag_query
    return params


def _load_pcm_f32(wav_path: str, sample_rate: int) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        if wf.getframerate() != sample_rate:
            raise ValueError(
                f"{wav_path} is {wf.getframerate()}Hz, but the live server expects {sample_rate}Hz "
                "(moshi.models.loaders.SAMPLE_RATE) -- resample the input WAV first."
            )
        raw = wf.readframes(wf.getnframes())
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    return (pcm_i16.astype(np.float32)) / 32768.0


async def run_streaming_query(
    server_url: str,
    voice_prompt: str,
    text_prompt: str,
    input_wav_path: str,
    output_wav_path: Optional[str] = None,
    output_text_path: Optional[str] = None,
    rag_query: str = "",
    seed: Optional[int] = None,
    realtime_pacing: bool = True,
    trailing_silence_s: float = 10.0,
    response_buffer_s: float = 15.0,
    connect_timeout_s: float = 30.0,
) -> dict:
    """Streams `input_wav_path` to the live server's `/api/chat` endpoint over a real WebSocket
    connection -- the same protocol the browser web UI speaks, not a re-run of `moshi.offline`.

    `realtime_pacing=True` (default) sends outgoing audio frames `FRAME_SIZE / SAMPLE_RATE`
    seconds apart, exactly like a real microphone would, instead of uploading the whole WAV as
    fast as the network allows -- this is what actually exercises "smooth real-time streaming"
    rather than just correctness. Trailing silence (`trailing_silence_s`) is appended after the
    WAV so the agent has room to finish speaking, then the client keeps listening for
    `response_buffer_s` more seconds before closing the connection -- mirroring the padded-WAV
    fix in docs/MODE_C_IMPLEMENTATION_REPORT.md Section 3c, just over a live socket instead of a
    fixed-duration `moshi.offline` run.

    Returns:
        {
          "transcript": str,
          "output_wav_path": str | None,
          "connect_latency_s": float | None,             # time to receive the handshake byte
          "first_text_token_latency_s": float | None,     # handshake -> first text token
          "total_duration_s": float,
        }
    """
    import aiohttp
    import sphn

    sample_rate, frame_size = _audio_constants()
    pcm = _load_pcm_f32(input_wav_path, sample_rate)
    frame_interval_s = frame_size / sample_rate
    wav_duration_s = len(pcm) / sample_rate
    total_deadline_s = wav_duration_s + trailing_silence_s + response_buffer_s

    params = build_query_params(voice_prompt, text_prompt, rag_query, seed)

    opus_writer = sphn.OpusStreamWriter(sample_rate)
    opus_reader = sphn.OpusStreamReader(sample_rate)

    transcript_parts: list[str] = []
    received_pcm: list[np.ndarray] = []
    timing = {"connect_latency_s": None, "first_text_token_latency_s": None}
    t_start = time.monotonic()

    async def send_audio(ws):
        offset = 0
        n_frames_total = (len(pcm) // frame_size) + 1
        silence = np.zeros(frame_size, dtype=np.float32)
        n_silence_frames = int(trailing_silence_s / frame_interval_s)
        for i in range(n_frames_total + n_silence_frames):
            if offset < len(pcm):
                chunk = pcm[offset: offset + frame_size]
                if len(chunk) < frame_size:
                    chunk = np.concatenate([chunk, np.zeros(frame_size - len(chunk), dtype=np.float32)])
                offset += frame_size
            else:
                chunk = silence
            opus_writer.append_pcm(chunk)
            data = opus_writer.read_bytes()
            if data:
                await ws.send_bytes(b"\x01" + data)
            if realtime_pacing:
                await asyncio.sleep(frame_interval_s)

    async def recv_responses(ws):
        async for message in ws:
            if message.type != aiohttp.WSMsgType.BINARY:
                continue
            data = message.data
            if not data:
                continue
            kind = data[0]
            if kind == 0:
                if timing["connect_latency_s"] is None:
                    timing["connect_latency_s"] = time.monotonic() - t_start
            elif kind == 1:
                opus_reader.append_bytes(data[1:])
                pcm_out = opus_reader.read_pcm()
                if pcm_out.shape[-1] > 0:
                    received_pcm.append(pcm_out)
            elif kind == 2:
                if timing["first_text_token_latency_s"] is None and timing["connect_latency_s"] is not None:
                    timing["first_text_token_latency_s"] = (
                        time.monotonic() - t_start - timing["connect_latency_s"]
                    )
                transcript_parts.append(data[1:].decode("utf-8", errors="replace"))

    timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout_s, sock_connect=connect_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(server_url, params=params, max_msg_size=0) as ws:
            send_task = asyncio.create_task(send_audio(ws))
            recv_task = asyncio.create_task(recv_responses(ws))
            try:
                await asyncio.wait_for(asyncio.gather(send_task, recv_task), timeout=total_deadline_s)
            except asyncio.TimeoutError:
                pass
            finally:
                for task in (send_task, recv_task):
                    if not task.done():
                        task.cancel()
                if not ws.closed:
                    await ws.close()

    total_duration_s = time.monotonic() - t_start
    transcript = "".join(transcript_parts)

    if received_pcm and output_wav_path:
        output_pcm = np.concatenate(received_pcm, axis=-1)
        sphn.write_wav(output_wav_path, output_pcm, sample_rate)
    elif not received_pcm:
        output_wav_path = None

    if output_text_path is not None:
        with open(output_text_path, "w") as f:
            json.dump(transcript_parts, f, ensure_ascii=False)

    return {
        "transcript": transcript,
        "output_wav_path": output_wav_path,
        "connect_latency_s": timing["connect_latency_s"],
        "first_text_token_latency_s": timing["first_text_token_latency_s"],
        "total_duration_s": total_duration_s,
    }
