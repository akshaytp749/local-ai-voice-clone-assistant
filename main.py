"""Entry point for the voice assistant.

Run without arguments to launch the web UI.
Pass --cli for a terminal-only session (--simple for non-streaming mode).
"""
import argparse
import logging
import os
import warnings

# Quieter Hugging Face / Whisper output.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*ffmpeg.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*", category=RuntimeWarning)

for _logger in ("transformers", "huggingface_hub", "httpx", "httpcore"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

from llm import ask, ask_streaming
from tts import speak
from voices import default_voice_id


def run_streaming(voice_id: str) -> None:
    """Stream LLM responses sentence-by-sentence and speak each as it arrives."""
    print("[Mode] Streaming - first sentence plays soon after typing Enter\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            speak("Goodbye!", voice_id)
            break

        print("Assistant: ", end="", flush=True)
        for sentence in ask_streaming(query):
            print(sentence, end=" ", flush=True)
            speak(sentence, voice_id)
        print()


def run_simple(voice_id: str) -> None:
    """Wait for the full LLM response before speaking it."""
    print("[Mode] Simple - full response generated before speaking\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            speak("Goodbye!", voice_id)
            break

        print("Thinking...")
        response = ask(query)
        print(f"Assistant: {response}\n")
        speak(response, voice_id)


def main() -> None:
    """Parse CLI arguments and either launch the web UI or a terminal session."""
    parser = argparse.ArgumentParser(description="Voice Clone + Local LLM Assistant")
    parser.add_argument("--cli", action="store_true", help="Use terminal UI instead of web UI")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use blocking (non-streaming) terminal mode",
    )
    args = parser.parse_args()

    if args.cli:
        voice_id = default_voice_id()
        if args.simple:
            run_simple(voice_id)
        else:
            run_streaming(voice_id)
        return

    from server import launch

    launch()


if __name__ == "__main__":
    main()
