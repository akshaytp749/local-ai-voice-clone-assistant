"""FastAPI server for the voice assistant.

Serves static files and exposes REST + WebSocket APIs for chat streaming,
voice management, TTS synthesis, and system status.
"""

import asyncio
import io
import os
import queue as _queue
import shutil
import tempfile
import threading
from pathlib import Path

import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    MAX_HISTORY_MESSAGES,
    OLLAMA_MODEL,
    OLLAMA_URL,
    TTS_FIRST_CHUNK_CHARS,
    TTS_MIN_CHUNK_CHARS,
    UI_HOST,
    UI_PORT,
)
from llm import ask_streaming, keep_warm
from tts import (
    clean_text,
    model_status,
    prepare_voice_steps,
    preload_voice,
    synthesize,
)
from voices import (
    add_voice,
    default_voice_id,
    list_voices,
    mark_voice_failed,
    preview_cache_path,
    remove_voice,
    resolve_voice_id,
    transcribe_audio_file,
    validate_voices,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Local AI Voice Clone Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Static page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page HUD frontend."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status() -> dict:
    """Return Ollama reachability, TTS engine state, voice count, and model name.

    Plain ``def`` on purpose: the blocking requests call runs in FastAPI's
    threadpool. As ``async def`` it froze the event loop for up to 2 s per
    poll, stalling WebSocket audio delivery (audible playback gaps).
    """
    try:
        resp = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        ollama_ok = resp.ok
    except Exception:
        ollama_ok = False

    return {
        "ollama": ollama_ok,
        "tts": model_status(),
        "voices": len(list_voices()),
        "model": OLLAMA_MODEL,
    }


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------

@app.get("/api/voices")
async def api_list_voices() -> list:
    """Return all configured voice profiles."""
    return list_voices()


@app.post("/api/voices")
def api_add_voice(
    name: str = Form(...),
    transcript: str = Form(""),
    file: UploadFile = File(...),
) -> dict:
    """Upload a voice sample, run preparation steps, and register the profile.

    Runs synchronously (FastAPI threadpool) because preparation can take
    30-90 seconds (Whisper transcription + TTS model warm-up).
    """
    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    voice_id: str | None = None
    try:
        voice_id = add_voice(name.strip(), tmp_path)
        for _ in prepare_voice_steps(voice_id, manual_ref_text=transcript.strip()):
            pass
        return {"id": voice_id, "name": name.strip(), "status": "ready"}
    except Exception as exc:
        if voice_id is not None:
            try:
                remove_voice(voice_id)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.delete("/api/voices/{voice_id}")
async def api_remove_voice(voice_id: str) -> dict:
    """Remove a voice profile by ID."""
    try:
        remove_voice(voice_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/voices/{voice_id}/preview")
def api_preview_voice(voice_id: str) -> Response:
    """Synthesise a short preview phrase and return it as a WAV audio response.

    The phrase is fixed per voice, so the result is cached on disk and repeat
    previews skip synthesis entirely.
    """
    resolved = resolve_voice_id(voice_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Voice not found")
    cache_file = preview_cache_path(resolved)
    if os.path.isfile(cache_file):
        return Response(content=Path(cache_file).read_bytes(), media_type="audio/wav")
    try:
        phrase = "Hello, I'm your voice assistant and I'm ready to help."
        audio, sr = synthesize(phrase, resolved)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        Path(cache_file).write_bytes(data)
        return Response(content=data, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/voices/{voice_id}/preload")
async def api_preload_voice(voice_id: str) -> dict:
    """Trigger background pre-warming of TTS reference audio for a voice."""
    resolved = resolve_voice_id(voice_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Voice not found")
    threading.Thread(target=preload_voice, args=(resolved,), daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Speech-to-text (mic input)
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
def api_transcribe(file: UploadFile = File(...)) -> dict:
    """Transcribe a browser mic recording to text using the local Whisper model.

    Runs synchronously (FastAPI threadpool); the Whisper pipeline is already
    resident in VRAM from voice setup, so this adds no extra model load.
    """
    suffix = Path(file.filename or "mic.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        text = transcribe_audio_file(tmp_path)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------

def _encode_wav(audio, sr: int) -> bytes:
    """Encode an audio array as 16-bit WAV bytes for binary WS transport.

    PCM_16 halves the payload vs float32 with no audible difference at 24 kHz,
    and binary frames avoid the ~33% base64 overhead plus the browser-side
    string decode loop.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim_history(history: list[dict[str, str]], max_messages: int) -> None:
    """Drop the oldest messages in place, in whole user/assistant pairs, so the
    list never exceeds ``max_messages`` and still starts with a user turn."""
    if len(history) > max_messages:
        excess = len(history) - max_messages
        excess += excess % 2
        del history[:excess]


def close_turn(history: list[dict[str, str]], response: str) -> None:
    """Record the assistant reply for the turn whose user message is last.

    A turn that produced nothing usable (LLM error, or stopped before the first
    sentence) removes its user message instead. Leaving it would put two user
    messages back to back on the next request, and the model would answer the
    failed question as well as the new one.
    """
    if response and not response.startswith("[Error]"):
        history.append({"role": "assistant", "content": response})
    elif history and history[-1]["role"] == "user":
        history.pop()


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Stream LLM sentences and per-sentence cloned-voice audio to the client.

    The browser plays the audio (Web Audio API), so synthesis streams sentence
    by sentence: sentence 1 can be speaking while the LLM is still writing
    sentence 3. Audio is sent as base64 WAV chunks in arrival order.

    Protocol (server -> client):
      binary frame: one cloned-voice chunk as 16-bit WAV bytes
      {"type": "token",    "text": "<sentence>"}      — LLM sentence arrived
      {"type": "done",     "full": "<response>"}      — full response complete
                                                        ("interrupted": true if cut short)
      {"type": "speaking", "state": true|false}       — TTS stream start/end
      {"type": "error",    "msg":  "<message>"}       — something went wrong

    Client -> server JSON:
      {"text": "...", "voice_id": "alice", "speak": true}
      {"type": "stop"}                                — stop the current reply

    Interruption: a message that arrives while a reply is still being generated
    cuts that reply short (the LLM stream is closed and queued speech is
    dropped). If it carries text, it becomes the next turn.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # Conversation memory for this connection.
    history: list[dict[str, str]] = []
    # Incremented on each user turn so stale TTS threads skip their work.
    tts_gen = 0
    # Serialise all sends: tokens (this coroutine) and audio (worker thread)
    # share one websocket, which is not safe under concurrent sends.
    send_lock = asyncio.Lock()

    async def _send(msg: dict) -> None:
        async with send_lock:
            await websocket.send_json(msg)

    def _send_threadsafe(msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(_send(msg), loop)

    async def _send_audio(data: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(data)

    def _send_audio_threadsafe(data: bytes) -> None:
        asyncio.run_coroutine_threadsafe(_send_audio(data), loop)

    # One receive is always pending, including while a reply is generated, so
    # a "stop" or a new question can interrupt it. A receive that did not fire
    # during a turn is kept and awaited at the top of the next loop.
    recv_task: asyncio.Task | None = None
    pending: dict | None = None  # message that interrupted the previous turn

    try:
        while True:
            if pending is not None:
                data, pending = pending, None
            else:
                if recv_task is None:
                    recv_task = asyncio.create_task(websocket.receive_json())
                data = await recv_task
                recv_task = None

            if data.get("type") == "stop":
                # Nothing is generating (that case is handled below), but
                # speech for the finished reply may still be synthesising.
                tts_gen += 1
                continue

            text: str = (data.get("text") or "").strip()
            voice_id: str = data.get("voice_id") or default_voice_id() or ""
            speak_enabled: bool = bool(data.get("speak", True))

            if not text:
                continue

            tts_gen += 1
            my_gen = tts_gen
            history.append({"role": "user", "content": text})

            # -- TTS worker: sentences -> cloned-voice audio chunks --------
            # Synthesis is GPU-bound and serial, so one worker thread pulls
            # sentences in order and streams each WAV to the browser.
            synth_q: _queue.Queue = _queue.Queue()
            speaking = speak_enabled and bool(voice_id)

            def _synth_worker(vid=voice_id, gen=my_gen) -> None:
                while True:
                    chunk = synth_q.get()
                    if chunk is None:
                        break
                    if gen != tts_gen:
                        continue  # a newer turn started — drain and skip
                    clean = clean_text(chunk)
                    if not clean:
                        continue
                    try:
                        audio, sr = synthesize(clean, vid)
                        if gen != tts_gen:
                            continue
                        _send_audio_threadsafe(_encode_wav(audio, sr))
                    except Exception as exc:
                        print(f"[Server] TTS error: {exc}", flush=True)
                if gen == tts_gen:
                    _send_threadsafe({"type": "speaking", "state": False})

            synth_thread: threading.Thread | None = None
            if speaking:
                await _send({"type": "speaking", "state": True})
                synth_thread = threading.Thread(target=_synth_worker, daemon=True)
                synth_thread.start()

            # -- Stream LLM sentences --------------------------------------
            # ask_streaming() is a blocking generator, so we run it in a
            # thread and feed sentences back to the async context via a queue.
            sentence_q: _queue.Queue = _queue.Queue()
            cancel = threading.Event()

            def _run_llm(msgs=list(history), cancel=cancel, out=sentence_q) -> None:
                gen = ask_streaming(msgs)
                try:
                    for sentence in gen:
                        if cancel.is_set():
                            break
                        s = sentence.strip()
                        if s:
                            out.put(s)
                except Exception as exc:
                    out.put(exc)
                finally:
                    gen.close()    # closes the HTTP stream, so Ollama stops generating
                    out.put(None)  # sentinel

            loop.run_in_executor(None, _run_llm)

            full_response = ""
            synth_buf = ""   # batches short sentences before handing to TTS
            first_chunk = True
            interrupted = False
            failed = False
            if recv_task is None:
                recv_task = asyncio.create_task(websocket.receive_json())
            while True:
                get_task = loop.run_in_executor(None, sentence_q.get)
                await asyncio.wait({get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
                if recv_task.done():
                    # The client spoke over the reply: stop the LLM, drop
                    # queued speech, and let the reader thread finish.
                    interrupted = True
                    cancel.set()
                    tts_gen += 1
                    while await get_task is not None:
                        get_task = loop.run_in_executor(None, sentence_q.get)
                    msg = recv_task.result()  # re-raises WebSocketDisconnect
                    recv_task = None
                    if (msg.get("text") or "").strip():
                        pending = msg
                    break
                item = get_task.result()
                if item is None:
                    break
                if isinstance(item, Exception):
                    failed = True
                    await _send({"type": "error", "msg": str(item)})
                    break
                full_response = (full_response + " " + item).strip()
                # UI text streams per sentence for a snappy transcript...
                await _send({"type": "token", "text": item})
                # ...but TTS receives batches so playback stays ahead of
                # synthesis. The first batch is small (fast first audio);
                # later batches are larger (no underrun gaps).
                if synth_thread is not None:
                    synth_buf = (synth_buf + " " + item).strip()
                    threshold = TTS_FIRST_CHUNK_CHARS if first_chunk else TTS_MIN_CHUNK_CHARS
                    if len(synth_buf) >= threshold:
                        synth_q.put(synth_buf)
                        synth_buf = ""
                        first_chunk = False

            # Flush any remainder, then close the queue; the worker emits
            # speaking:false when drained.
            if synth_thread is not None:
                if synth_buf and not interrupted:
                    synth_q.put(synth_buf)
                synth_q.put(None)

            done_msg: dict = {"type": "done", "full": full_response}
            if interrupted:
                done_msg["interrupted"] = True
            await _send(done_msg)

            # -- Update conversation memory --------------------------------
            # An interrupted reply keeps what was said so far; a failed or
            # empty turn is removed entirely (see close_turn).
            close_turn(history, "" if failed else full_response)
            trim_history(history, MAX_HISTORY_MESSAGES)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await _send({"type": "error", "msg": str(exc)})
        except Exception:
            pass
    finally:
        tts_gen += 1  # connection is gone: let any synth worker drain and exit
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup_warmup() -> None:
    """Pre-load all voices and warm up Ollama before the first user request."""
    validate_voices()
    for voice in list_voices():
        try:
            preload_voice(voice["id"])
        except Exception as exc:
            mark_voice_failed(voice["id"], str(exc))
            print(f"[Startup] Voice '{voice['name']}' skipped: {exc}", flush=True)

    print("[LLM] Warming up Ollama model...", flush=True)
    keep_warm()


def launch() -> None:
    """Validate voices, start background warmup, and launch the uvicorn server."""
    threading.Thread(target=_startup_warmup, daemon=True).start()
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="warning")
