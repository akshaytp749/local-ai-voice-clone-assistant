"""WebSocket chat protocol: memory, failed turns, and interruption.

The heavy modules (tts -> torch/F5-TTS, voices -> Whisper) are replaced with
stubs before server.py is imported, so this runs without a GPU or any model.
"""
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tts = types.ModuleType("tts")
_tts.clean_text = lambda text: text
_tts.model_status = lambda: "ready"
_tts.prepare_voice_steps = lambda *a, **k: iter(())
_tts.preload_voice = lambda *a, **k: None
_tts.synthesize = lambda text, voice_id: ([0.0] * 240, 24000)
sys.modules["tts"] = _tts

_voices = types.ModuleType("voices")
for _name in (
    "add_voice", "mark_voice_failed", "preview_cache_path", "remove_voice",
    "transcribe_audio_file", "validate_voices",
):
    setattr(_voices, _name, lambda *a, **k: None)
_voices.default_voice_id = lambda: ""
_voices.list_voices = lambda: []
_voices.resolve_voice_id = lambda v: v
sys.modules["voices"] = _voices

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    """Record the message list each LLM call receives; reply from a script."""
    seen: list[list[dict]] = []
    script: list = []

    def fake_stream(messages):
        seen.append([dict(m) for m in messages])
        reply = script.pop(0) if script else ["Okay."]
        if isinstance(reply, Exception):
            raise reply
        yield from reply

    monkeypatch.setattr(server, "ask_streaming", fake_stream)
    return types.SimpleNamespace(seen=seen, script=script)


def read_turn(ws):
    """Collect JSON messages up to and including 'done'."""
    out = []
    while True:
        msg = ws.receive_json()
        out.append(msg)
        if msg["type"] == "done":
            return out


def ask(ws, text):
    ws.send_json({"text": text, "speak": False})
    return read_turn(ws)


def roles(messages):
    return [m["role"] for m in messages]


def test_streams_tokens_then_done(calls):
    calls.script.append(["Hello.", "How are you?"])
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        msgs = ask(ws, "hi")
    assert [m["text"] for m in msgs if m["type"] == "token"] == ["Hello.", "How are you?"]
    assert msgs[-1] == {"type": "done", "full": "Hello. How are you?"}


def test_second_turn_sees_first_turn_in_history(calls):
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        ask(ws, "first")
        ask(ws, "second")
    assert roles(calls.seen[1]) == ["user", "assistant", "user"]
    assert calls.seen[1][-1]["content"] == "second"


def test_failed_turn_does_not_leave_a_dangling_user_message(calls):
    calls.script.append(["[Error] Cannot reach Ollama. Start it with: ollama serve"])
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        ask(ws, "lost question")
        ask(ws, "next question")
    # Before the fix this was ["user", "user"]: the failed question stayed.
    assert roles(calls.seen[1]) == ["user"]
    assert calls.seen[1][0]["content"] == "next question"


def test_llm_exception_reports_error_and_drops_the_turn(calls):
    calls.script.append(RuntimeError("model crashed"))
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        msgs = ask(ws, "boom")
        ask(ws, "again")
    assert any(m["type"] == "error" and "model crashed" in m["msg"] for m in msgs)
    assert roles(calls.seen[1]) == ["user"]


def test_stop_interrupts_a_reply_in_progress(monkeypatch):
    release = threading.Event()
    closed = threading.Event()

    def slow_stream(messages):
        try:
            yield "First sentence."
            release.wait(5)          # the model is "still writing" here
            yield "Second sentence."
            yield "Third sentence."
        finally:
            closed.set()             # generator closed -> HTTP stream closed

    monkeypatch.setattr(server, "ask_streaming", slow_stream)
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        ws.send_json({"text": "tell me a long story", "speak": False})
        assert ws.receive_json() == {"type": "token", "text": "First sentence."}
        ws.send_json({"type": "stop"})
        release.set()
        msgs = read_turn(ws)
    done = msgs[-1]
    assert done["interrupted"] is True
    assert "Third sentence." not in done["full"]
    assert closed.wait(5)


def test_new_question_interrupts_and_is_answered_next(monkeypatch):
    release = threading.Event()
    seen = []

    def stream(messages):
        seen.append([dict(m) for m in messages])
        if len(seen) == 1:
            yield "Long answer part one."
            release.wait(5)
            yield "Long answer part two."
        else:
            yield "Short answer."

    monkeypatch.setattr(server, "ask_streaming", stream)
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        ws.send_json({"text": "explain everything", "speak": False})
        assert ws.receive_json()["text"] == "Long answer part one."
        ws.send_json({"text": "never mind, what time is it?", "speak": False})
        release.set()
        first = read_turn(ws)
        second = read_turn(ws)
    assert first[-1]["interrupted"] is True
    assert second[-1] == {"type": "done", "full": "Short answer."}
    # The partial reply is kept, so roles still alternate for the model.
    assert roles(seen[1]) == ["user", "assistant", "user"]
    assert seen[1][1]["content"] == "Long answer part one."


def test_stop_while_idle_is_harmless(calls):
    with TestClient(server.app).websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "stop"})
        msgs = ask(ws, "hello")
    assert msgs[-1]["type"] == "done"
    assert len(calls.seen) == 1


class TestHistoryHelpers:
    def test_trim_drops_whole_pairs(self):
        history = [{"role": r, "content": str(i)} for i, r in enumerate(["user", "assistant"] * 4)]
        server.trim_history(history, 5)
        assert len(history) == 4
        assert history[0]["role"] == "user"

    def test_trim_noop_under_limit(self):
        history = [{"role": "user", "content": "a"}]
        server.trim_history(history, 10)
        assert len(history) == 1

    def test_close_turn_appends_reply(self):
        history = [{"role": "user", "content": "q"}]
        server.close_turn(history, "a")
        assert roles(history) == ["user", "assistant"]

    def test_close_turn_removes_failed_question(self):
        history = [{"role": "user", "content": "q"}]
        server.close_turn(history, "")
        assert history == []
