# Local AI Voice Clone Assistant

A private, fully-local voice assistant powered by Ollama and F5-TTS zero-shot
voice cloning, with a custom sci-fi HUD web UI (FastAPI + WebSocket). Talk to
it with your mic and it replies in any voice you've cloned, all on your own
machine. Nothing leaves the box.

| Layer | Tool |
|---|---|
| LLM | Ollama + `qwen3.5:2b` (multi-turn chat with memory) |
| Voice cloning / TTS | F5-TTS |
| Speech-to-text (mic + reference) | Whisper large-v3-turbo |
| Audio | browser Web Audio API, `soundfile`, `imageio-ffmpeg` |
| Web UI | FastAPI + `uvicorn`, vanilla HTML/CSS/JS HUD |
| Language | Python 3.11 (recommended) |

Target hardware: a CUDA-capable NVIDIA GPU is strongly recommended (developed
on an RTX 4060 Laptop, 8 GB VRAM). CPU mode works but voice generation will be
much slower.

## Features

- **Zero-shot voice cloning** - upload any 30-60 second recording; F5-TTS clones it immediately with no fine-tuning.
- **Talk to it** - click the mic, speak, and your words are transcribed locally by Whisper (audio never leaves your machine, no cloud speech API).
- **Cloned voice plays in the browser** - TTS audio streams to the client over WebSocket and plays via the Web Audio API, so it works from a phone or another device on your network, not just the server's speakers.
- **Sentence-streaming TTS** - each sentence is synthesised and spoken as soon as it's ready while the LLM keeps writing, so the first words start fast.
- **Interruptible** - press `Esc` or click the orb to cut a reply short; the LLM stream is closed and queued speech is dropped, so the next question starts immediately.
- **Conversation memory** - the assistant remembers the last several turns (Ollama `/api/chat`).
- **Voice preloading** - switching voices triggers background preload so the first reply is near-instant.
- **Reference audio auto-trimming** - clips longer than 10 seconds are trimmed automatically because F5-TTS produces garbled output beyond that limit.
- **LLM warmup at startup** - Ollama is pinged at launch so `qwen3.5:2b` is already in VRAM when the first message arrives.
- **CLI fallback** - `--cli` streams sentence-by-sentence to the terminal; `--simple` waits for the full response.

## Quick Start

### 1. One-time environment setup

```powershell
cd local-ai-voice-clone-assistant
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

Install PyTorch with CUDA separately (adjust the CUDA version to match your driver):

```powershell
.\venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Pull the LLM model

```powershell
ollama pull qwen3.5:2b
```

### 3. Run

```powershell
# Terminal 1: start Ollama
ollama serve

# Terminal 2: start the assistant
.\run.ps1
```

Open the URL shown in the terminal. The default is `http://127.0.0.1:8000`.

### 4. Add a voice

In the **Voice Studio** panel:

1. Enter a voice name.
2. Upload or record a 30-60 second clean speech sample.
3. Optionally paste the transcript of the recording to skip automatic Whisper transcription.
4. Click **Add**.

Once the status badge changes to **Ready**, select the voice in the dropdown and enable **Speak responses aloud**.

## How Voices Work

F5-TTS uses *zero-shot voice cloning*: given a short reference clip and its transcript, it synthesises new speech that matches the speaker's timbre and prosody with no training required.

**Key constraints:**
- Reference clips longer than **10 seconds** cause garbled / gibberish output. The app trims them automatically.
- Clip quality matters: clean, low-noise recordings produce significantly better results.
- The transcript must match the recording exactly. Mismatches degrade quality.

Whisper `large-v3-turbo` transcribes the reference clip automatically when no manual transcript is provided. You can paste the transcript yourself to skip this step and avoid the one-time model download.

## Project Structure

```text
local-ai-voice-clone-assistant/
|-- server.py           # FastAPI app: static files, REST + WebSocket chat
|-- main.py             # Entry point (web and CLI modes)
|-- llm.py              # Ollama chat wrapper (streaming + history)
|-- tts.py              # F5-TTS wrapper and synthesis pipeline
|-- voices.py           # Voice registry, audio prep, Whisper transcription
|-- config.py           # App settings
|-- static/             # HUD frontend (index.html, style.css, app.js, orb.js)
|-- run.ps1             # Windows launcher
|-- requirements.txt
`-- voices/             # Voice samples, manifest (voices.json), and WAV cache (created at runtime, not committed)
```

## CLI Modes

```powershell
.\run.ps1 --cli           # streaming: speaks each sentence as it arrives
.\run.ps1 --cli --simple  # simple: waits for full response before speaking
```

## Configuration

Edit `config.py` to change any of the following:

`OLLAMA_MODEL`, `OLLAMA_HOST`, `TTS_NFE_STEP`, `UI_HOST` and `UI_PORT` can also be set as environment variables of the same name, without editing the file.

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:2b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `TTS_SPEED` | `1.0` | Synthesis speed (try `0.9` if audio crackles) |
| `TTS_NFE_STEP` | `16` | **Latency vs quality knob.** 16 is about 2x faster than real-time (seamless, low latency) on an RTX 4060; raise to 24/32 for higher quality at the cost of latency and possible inter-sentence gaps. |
| `TTS_MIN_CHUNK_CHARS` | `40` | Min characters per TTS batch (larger = fewer seams, more latency) |
| `TTS_FIRST_CHUNK_CHARS` | `24` | First batch size - smaller starts audio sooner |
| `TARGET_SR` | `24000` | Prepared reference audio sample rate |
| `UI_HOST` | `127.0.0.1` | Web server host (use `0.0.0.0` to reach it from other devices) |
| `UI_PORT` | `8000` | Web server port |
| `MAX_HISTORY_MESSAGES` | `10` | Conversation turns kept as memory |

## Troubleshooting

| Problem | Fix |
|---|---|
| Ollama is not connected | Run `ollama serve`, then refresh the app. |
| The model is missing | Run `ollama pull qwen3.5:2b`. |
| Voice setup fails during transcription | Paste the transcript in the optional field and try again. |
| Mic button does nothing / permission denied | Browsers only allow microphone access on a secure origin. Use `http://127.0.0.1:8000` (localhost counts as secure) and allow mic access when prompted. |
| No sound from replies | Click anywhere first (browsers block audio until a user gesture), confirm a voice is selected, and that the **VOICE** toggle is on. |
| Voice generation is slow | Confirm CUDA works: `venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`. |
| Audio crackles | Try `TTS_SPEED = 0.9` in `config.py`. |
| A saved voice shows an error | Remove it and add the voice sample again. |
| `UnicodeEncodeError` / `cp1252` crash | Run the assistant in a terminal that supports UTF-8 (`chcp 65001`). |
| Reference audio sounds garbled | The clip may be over 10 seconds; the app trims it automatically, but very long clips can still confuse the model. Use a 30-60 second recording. |
