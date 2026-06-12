"""Ollama LLM client for single-shot and streaming chat completions.

Uses Ollama's /api/chat endpoint so the assistant has real conversation
memory: callers pass a list of {"role", "content"} messages. For backward
compatibility (CLI mode), a bare string is accepted and treated as a single
user turn with no history.

Exposes ask(), ask_streaming(), and keep_warm().
"""

import json

import requests

from config import (
    OLLAMA_CHAT_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_URL,
    SENTENCE_END_CHARS,
    SYSTEM_PROMPT,
)

Messages = list[dict[str, str]]

# Reuse one TCP connection to Ollama instead of reconnecting every request.
_session = requests.Session()


def _to_messages(prompt: "str | Messages") -> Messages:
    """Normalise a bare string into a single-turn message list.

    Strings come from CLI mode (stateless); the web server passes a full
    history list directly.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def _with_system(messages: Messages) -> Messages:
    """Prepend the system prompt unless the caller already supplied one."""
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def ask(prompt: "str | Messages") -> str:
    """Send a prompt (or message history) and return the full response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _with_system(_to_messages(prompt)),
        "stream": False,
        # Qwen3-family models are hybrid reasoners; thinking tokens would add
        # seconds of silence before the first spoken sentence.
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }
    try:
        response = _session.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot reach Ollama. Start it with: ollama serve"
    except Exception as exc:
        return f"[Error] {exc}"


def keep_warm() -> None:
    """Load the Ollama model into VRAM at startup so first response is instant."""
    try:
        _session.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": " ",  # blank prompt keeps model in VRAM
                  "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE, "options": OLLAMA_OPTIONS},
            timeout=180,
        )
        print(f"[LLM] Model '{OLLAMA_MODEL}' loaded and warm.", flush=True)
    except Exception as exc:
        print(f"[LLM] Warm-up skipped: {exc}", flush=True)


# Clause delimiters let us flush sub-sentence chunks so the first audio
# starts sooner. We only break on a clause once the buffer is long enough
# to be worth synthesising on its own.
_CLAUSE_END = (",", ";", ":", "—", "–")
_CLAUSE_MIN_CHARS = 24
# Hard cap: a long comma-less run is force-flushed at a word boundary so no
# single TTS chunk becomes huge (which would block first audio for seconds).
_MAX_CHUNK_CHARS = 48

# Periods that don't end a sentence: decimals ("3.14") and common
# abbreviations ("Dr.") would otherwise flush mid-token and read awkwardly.
_ABBREVIATIONS = frozenset((
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "no.", "approx.", "dept.", "inc.", "jr.", "sr.",
))


def _ends_sentence(stripped: str) -> bool:
    """True when the buffer ends at a real sentence boundary."""
    if not stripped.endswith(SENTENCE_END_CHARS):
        return False
    if stripped.endswith(".") and not stripped.endswith(("...", "!", "?")):
        if len(stripped) >= 2 and stripped[-2].isdigit():
            return False  # likely a decimal still being streamed
        last_word = stripped.rsplit(None, 1)[-1].lower()
        if last_word in _ABBREVIATIONS:
            return False
    return True


def ask_streaming(prompt: "str | Messages"):
    """Yield text chunks from Ollama's streaming chat API.

    Flushes on sentence ends and on clause boundaries (once long enough),
    so downstream TTS can begin speaking before the whole sentence exists.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _with_system(_to_messages(prompt)),
        "stream": True,
        "think": False,  # see ask(): avoid silent reasoning before first audio
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }
    try:
        with _session.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                buffer += data.get("message", {}).get("content", "")
                stripped = buffer.rstrip()

                end_sentence = _ends_sentence(stripped)
                end_clause = stripped.endswith(_CLAUSE_END) and len(stripped) >= _CLAUSE_MIN_CHARS

                if end_sentence or end_clause:
                    chunk = buffer.strip()
                    if chunk:
                        yield chunk
                    buffer = ""
                elif len(stripped) >= _MAX_CHUNK_CHARS:
                    # No delimiter yet — flush everything up to the last word
                    # boundary so we never emit a giant chunk, keep the rest.
                    cut = buffer.rstrip().rfind(" ")
                    if cut > 0:
                        chunk = buffer[:cut].strip()
                        if chunk:
                            yield chunk
                        buffer = buffer[cut + 1:]

                if data.get("done"):
                    break

            if buffer.strip():
                yield buffer.strip()

    except requests.exceptions.ConnectionError:
        yield "[Error] Cannot reach Ollama. Start it with: ollama serve"
    except Exception as exc:
        yield f"[Error] {exc}"
