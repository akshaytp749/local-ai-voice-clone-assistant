"""Sentence/clause chunking of the streamed LLM reply (llm.ask_streaming).

No Ollama needed: the HTTP session is replaced with a fake that replays
token-by-token NDJSON lines the way Ollama's /api/chat stream does.
"""
import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm  # noqa: E402


class FakeResponse:
    def __init__(self, tokens):
        self._tokens = tokens
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for i, tok in enumerate(self._tokens):
            done = i == len(self._tokens) - 1
            yield json.dumps({"message": {"content": tok}, "done": done}).encode()


def stream(monkeypatch, tokens):
    response = FakeResponse(tokens)
    monkeypatch.setattr(llm._session, "post", lambda *a, **k: response)
    return response


def words(text):
    """Split into word tokens with leading spaces, like a real token stream."""
    parts = text.split(" ")
    return [parts[0]] + [" " + p for p in parts[1:]]


class TestEndsSentence:
    @pytest.mark.parametrize("text", ["Hello there.", "Really?", "Stop!", "Well..."])
    def test_real_endings(self, text):
        assert llm._ends_sentence(text)

    @pytest.mark.parametrize("text", ["Pi is 3.", "Ask Dr.", "Fruit, e.g.", "No ending", "Version 2."])
    def test_not_endings(self, text):
        assert not llm._ends_sentence(text)


class TestAskStreaming:
    def test_splits_on_sentences(self, monkeypatch):
        stream(monkeypatch, words("Hi there. How are you?"))
        assert list(llm.ask_streaming("x")) == ["Hi there.", "How are you?"]

    def test_decimal_is_not_split(self, monkeypatch):
        stream(monkeypatch, ["Pi", " is", " 3.", "14", " roughly."])
        assert list(llm.ask_streaming("x")) == ["Pi is 3.14 roughly."]

    def test_abbreviation_is_not_split(self, monkeypatch):
        stream(monkeypatch, words("Ask Dr. Smith now."))
        assert list(llm.ask_streaming("x")) == ["Ask Dr. Smith now."]

    def test_long_clause_flushes_at_comma(self, monkeypatch):
        stream(monkeypatch, words("When the weather is nice outside, we walk."))
        chunks = list(llm.ask_streaming("x"))
        assert chunks[0] == "When the weather is nice outside,"
        assert " ".join(chunks) == "When the weather is nice outside, we walk."

    def test_long_run_without_punctuation_is_capped(self, monkeypatch):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        stream(monkeypatch, words(text))
        chunks = list(llm.ask_streaming("x"))
        assert len(chunks) > 1
        assert all(len(c) <= llm._MAX_CHUNK_CHARS + 12 for c in chunks)
        assert " ".join(chunks) == text  # nothing lost or duplicated

    def test_trailing_text_without_punctuation_is_flushed(self, monkeypatch):
        stream(monkeypatch, words("Done. And then"))
        assert list(llm.ask_streaming("x")) == ["Done.", "And then"]

    def test_closing_the_generator_closes_the_http_stream(self, monkeypatch):
        # This is what makes interruption stop Ollama generating.
        response = stream(monkeypatch, words("First. Second. Third."))
        gen = llm.ask_streaming("x")
        assert next(gen) == "First."
        gen.close()
        assert response.closed

    def test_connection_error_becomes_error_chunk(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(llm._session, "post", boom)
        (chunk,) = list(llm.ask_streaming("x"))
        assert chunk.startswith("[Error]")


class TestMessages:
    def test_string_prompt_gets_system_prompt(self):
        msgs = llm._with_system(llm._to_messages("hello"))
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_existing_system_prompt_is_kept(self):
        msgs = [{"role": "system", "content": "custom"}, {"role": "user", "content": "hi"}]
        assert llm._with_system(msgs) == msgs
