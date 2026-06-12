"""Central configuration for the voice assistant.

All tuneable settings live here so they can be changed without touching
application logic. Import individual constants where needed.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LLM
OLLAMA_MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Keep the model resident in VRAM between requests. Without this Ollama
# unloads after its 5-minute default and the next question pays a full
# multi-second model reload mid-conversation.
OLLAMA_KEEP_ALIVE = "30m"

# How many recent messages (user + assistant) to keep as conversation memory.
# 10 = the last 5 exchanges. Kept small so the context window never overflows.
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = (
    "You are a warm, friendly voice assistant with personality. "
    "Speak naturally like a real person: use contractions (I'm, you're, that's, it's), "
    "vary your sentence rhythm, and show genuine enthusiasm, curiosity, or empathy where fitting. "
    "Be concise, 1 to 2 sentences. "
    "Never use markdown, bullet points, lists, or any special characters. "
    "Write exactly as you would speak aloud."
)

OLLAMA_OPTIONS = {
    "num_gpu": 99,
    # 4096 is cheap for a 2B model and avoids silent top-truncation (which
    # eats the system prompt first) on longer conversations.
    "num_ctx": 4096,
    "temperature": 0.7,
}

# TTS
VOICES_DIR = os.path.join(BASE_DIR, "voices")
VOICES_MANIFEST = os.path.join(VOICES_DIR, "voices.json")
TTS_SPEED = 1.0
TARGET_SR = 24000

# Minimum characters before a batch of streamed text is sent to TTS.
# Tiny sentences ("Sure!") synthesise slower than they play and cause
# audible gaps; batching them to this size keeps playback ahead of synthesis
# while still starting audio early. UI text still streams per-sentence.
TTS_MIN_CHUNK_CHARS = 40
# The very first chunk uses a smaller threshold so the first words start
# sooner (lower response latency); later chunks use the larger size above.
TTS_FIRST_CHUNK_CHARS = 24

# F5-TTS Flow-Matching steps — the main latency knob. Synthesis time scales
# almost linearly with this. On an RTX 4060, 32 runs ~1.9x slower than
# real-time; 24 is a good balance (~25% faster, usually imperceptible);
# 16 is near real-time but audibly lower quality. Raise to 32 for best audio.
TTS_NFE_STEP = 16

# UI
UI_HOST = "127.0.0.1"
UI_PORT = 8000

# Sentence detection
SENTENCE_END_CHARS = (".", "!", "?", "...")
