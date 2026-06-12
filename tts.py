"""F5-TTS voice cloning wrapper with GPU acceleration.

Provides model loading, audio synthesis, reference-audio trimming,
an async two-stage synthesize-then-play pipeline, and speaker preloading.
"""

import contextlib
import importlib.machinery
import io
import os
import queue
import re as _re
import sys
import threading
import types
from collections.abc import Iterator

import numpy as np

# ===============================================
#  WORKAROUNDS — must run before all other imports
# ===============================================

try:
    import shutil as _shutil
    import tempfile as _tempfile
    import imageio_ffmpeg as _ioff

    _ffmpeg_exe = _ioff.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)

    # imageio_ffmpeg bundles ffmpeg as e.g. "ffmpeg-win-x86_64-v7.1.exe".
    # torchaudio (and pydub) call shutil.which("ffmpeg") which only matches
    # "ffmpeg.exe" exactly on Windows — so they can't find the bundled binary.
    # Fix: copy the exe to a stable temp path under the plain name
    # "ffmpeg.exe" (reused across launches so we don't litter temp dirs).
    _ffmpeg_alias_dir = os.path.join(_tempfile.gettempdir(), "voice_clone_ffmpeg")
    _ffmpeg_alias = os.path.join(_ffmpeg_alias_dir, "ffmpeg.exe")
    os.makedirs(_ffmpeg_alias_dir, exist_ok=True)
    if (not os.path.isfile(_ffmpeg_alias)
            or os.path.getsize(_ffmpeg_alias) != os.path.getsize(_ffmpeg_exe)):
        _shutil.copy2(_ffmpeg_exe, _ffmpeg_alias)

    os.environ["PATH"] = _ffmpeg_alias_dir + os.pathsep + _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = _ffmpeg_exe
    os.environ["IMAGEIO_FFMPEG_EXE"] = _ffmpeg_exe
except ImportError:
    pass

class _NullModule(types.ModuleType):
    """Permissive stub: any attribute access returns None instead of AttributeError."""
    __path__ = []
    __version__ = "1.3.2"

    def __getattr__(self, name: str):
        return None

_sk = _NullModule("sklearn")
_sk.__spec__ = importlib.machinery.ModuleSpec("sklearn", None, is_package=True)
sys.modules["sklearn"] = _sk

_skm = _NullModule("sklearn.metrics")
sys.modules["sklearn.metrics"] = _skm

# ===============================================
#  Normal imports
# ===============================================

import sounddevice as sd
import soundfile as sf
import tempfile

try:
    import torch
except ImportError as e:
    print(f"[TTS] Missing dependency: {e}")
    sys.exit(1)

from config import TTS_NFE_STEP, TTS_SPEED
from voices import (
    default_voice_id,
    get_voice_ref_text,
    mark_voice_ready,
    prepare_reference_audio,
    resolve_voice_id,
    transcribe_reference_text,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TTS] Using device: {device.upper()}")

if device == "cuda":
    # TF32 matmul on Ampere+/Ada is a free flow-matching speedup with no
    # audible quality change; benchmark mode picks the fastest conv kernels.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# F5-TTS concatenates ref_audio + gen_audio internally.
# Reference clips longer than this cause garbled / gibberish output.
MAX_REF_SECONDS = 10

# (original_path, ref_text) -> (trimmed_path, trimmed_text). Fully memoised:
# synthesize() calls this per streamed chunk, so the hot path must not touch
# the disk (the un-cached version cost two WAV reads per sentence).
_trimmed_ref_cache: dict[tuple[str, str], tuple[str, str]] = {}
_trimmed_ref_lock = threading.Lock()


def _trim_ref(ref_audio_path: str, ref_text: str) -> tuple[str, str]:
    """
    Trim reference audio to MAX_REF_SECONDS and shorten ref_text proportionally.
    Returns (trimmed_wav_path, trimmed_ref_text).
    """
    key = (ref_audio_path, ref_text)
    cached = _trimmed_ref_cache.get(key)
    if cached is not None:
        return cached

    with _trimmed_ref_lock:
        cached = _trimmed_ref_cache.get(key)
        if cached is not None:
            return cached

        orig_data, orig_sr = sf.read(ref_audio_path)
        orig_secs = len(orig_data) / orig_sr

        if orig_secs <= MAX_REF_SECONDS:
            result = (ref_audio_path, ref_text)
        else:
            max_samples = int(MAX_REF_SECONDS * orig_sr)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, orig_data[:max_samples], orig_sr)
            tmp.close()
            # Trim ref_text proportionally by word count
            words = ref_text.split()
            ratio = MAX_REF_SECONDS / orig_secs
            keep = max(1, int(len(words) * ratio))
            result = (tmp.name, " ".join(words[:keep]))
            print(f"[TTS] Trimmed reference audio from {orig_secs:.1f}s -> {MAX_REF_SECONDS}s", flush=True)

        _trimmed_ref_cache[key] = result
        return result


_tts_model = None
_model_status = "not_loaded"
_tts_lock = threading.Lock()


def _silent(*_args, **_kwargs):
    pass


@contextlib.contextmanager
def _quiet_f5():
    """Suppress noisy F5-TTS console output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def model_status() -> str:
    return _model_status


def warmup() -> None:
    global _model_status
    if _model_status == "ready":
        return
    _model_status = "loading"
    try:
        _get_model()
        _model_status = "ready"
    except Exception as e:
        _model_status = "error"
        raise RuntimeError(f"Failed to load TTS model: {e}") from e


def _get_model():
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    with _tts_lock:
        if _tts_model is None:
            print("[TTS] Loading F5-TTS voice cloning model...", flush=True)
            from f5_tts.api import F5TTS

            with _quiet_f5():
                _tts_model = F5TTS(device=device)
            print("[TTS] Model loaded!", flush=True)
    return _tts_model


def _normalize_audio(audio) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1e-8:
        arr = arr / peak
        arr = np.clip(arr, -0.98, 0.98)
    return arr


def _trim_edge_silence(audio: np.ndarray, sr: int, thresh: float = 0.015,
                       margin_ms: int = 18) -> np.ndarray:
    """Trim leading/trailing near-silence so streamed chunks butt together
    seamlessly. A small margin is kept so consonants aren't clipped."""
    if audio.size == 0:
        return audio
    above = np.where(np.abs(audio) > thresh)[0]
    if above.size == 0:
        return audio
    margin = int(sr * margin_ms / 1000)
    start = max(0, above[0] - margin)
    end = min(len(audio), above[-1] + margin)
    return audio[start:end]


def synthesize(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    voice_id = resolve_voice_id(voice_id)
    tts = _get_model()
    ref_audio = prepare_reference_audio(voice_id)
    ref_text = get_voice_ref_text(voice_id)
    ref_audio, ref_text = _trim_ref(ref_audio, ref_text)

    with _quiet_f5():
        audio, sr, _ = tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            speed=TTS_SPEED,
            show_info=_silent,
            progress=None,
            # -- Speed note -------------------------------
            # F5-TTS uses Flow Matching. Fewer steps = faster synthesis (lower
            # latency) but lower quality; 32 is the safe default (see config).
            nfe_step=TTS_NFE_STEP,
            sway_sampling_coef=-1,
            cfg_strength=2,
            # remove_silence is intentionally off: F5 implements it by writing
            # a temp WAV and running pydub per chunk; _trim_edge_silence below
            # does the same job in-memory.
            remove_silence=False,
        )

    audio = _normalize_audio(audio)
    # Trim silence at both edges so consecutive streamed chunks join cleanly.
    # (No leading-silence pad — the browser's Web Audio scheduler handles the
    # cold start, and any pad would become an audible gap between sentences.)
    audio = _trim_edge_silence(audio, sr)

    return audio, sr


def preload_voice(voice_id: str) -> None:
    """
    Pre-warm the TTS model and load reference audio for `voice_id`
    so the first real synthesis call is near-instant.
    Call this in a background thread whenever the user switches voices.
    """
    try:
        voice_id = resolve_voice_id(voice_id)
        if not voice_id:
            return
        warmup()                        # no-op if already loaded
        prepare_reference_audio(voice_id)   # converts + caches WAV
        print(f"[TTS] Voice '{voice_id}' preloaded and ready.", flush=True)
    except Exception as exc:
        print(f"[TTS] Preload warning for '{voice_id}': {exc}", flush=True)


def prepare_voice_steps(voice_id: str, test_phrase: str = "Hello, my voice is ready.", manual_ref_text: str = "") -> Iterator[str]:
    voice_id = resolve_voice_id(voice_id)
    yield "Converting audio to the right format..."
    prepare_reference_audio(voice_id)
    if manual_ref_text.strip() or not get_voice_ref_text(voice_id):
        yield "Transcribing your voice sample..." if not manual_ref_text.strip() else "Saving transcript..."
    transcribe_reference_text(voice_id, manual_text=manual_ref_text)
    yield "Loading TTS model - first time can take 1-2 minutes..."
    warmup()
    yield "Running a quick voice test..."
    synthesize(test_phrase, voice_id)
    mark_voice_ready(voice_id)
    yield "Voice ready."


def _play_audio(audio: np.ndarray, samplerate: int) -> None:
    try:
        # Do NOT call sd.stop() here — in serial queue mode it would cut
        # the last milliseconds of the previous sentence still in the buffer.
        sd.play(audio, samplerate=samplerate)
        sd.wait()
    except Exception as exc:
        print(f"[TTS] Playback error: {exc}")


# -- Two-stage pipeline: synthesize ahead while playing ------------------------
#
#  Stage 1 — _synth_worker: text -> audio  (runs on GPU, ~2s per sentence)
#  Stage 2 — _play_worker:  audio -> speakers (blocks for audio duration)
#
#  While stage 2 plays sentence N, stage 1 synthesizes sentence N+1 so
#  there is ZERO silence gap between consecutive sentences.

_synth_queue: queue.Queue = queue.Queue()        # (text, voice_id)
_play_queue:  queue.Queue = queue.Queue(maxsize=2)  # (audio, sr)  — bounded
_pipeline_started = False
_pipeline_lock = threading.Lock()
_stop_event = threading.Event()


def _clean_for_tts(text: str) -> str:
    """Strip markdown and special chars that confuse the TTS model."""
    text = _re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)   # **bold** / *italic*
    text = _re.sub(r"`{1,3}[^`]*`{1,3}", "", text)           # `code`
    text = _re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)   # [link](url)
    text = _re.sub(r"#{1,6}\s*", "", text)                    # # headings
    text = _re.sub(r"[-–—]{2,}", ",", text)                   # --- dashes
    text = _re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)          # other symbols
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _synth_worker() -> None:
    """Stage 1: convert text sentences to audio arrays."""
    while not _stop_event.is_set():
        try:
            item = _synth_queue.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:
            _play_queue.put(None)   # propagate sentinel
            _synth_queue.task_done()
            break
        text, voice_id = item
        try:
            clean = _clean_for_tts(text)
            if clean:
                audio, sr = synthesize(clean, voice_id)
                _play_queue.put((audio, sr))   # blocks if play_queue is full (back-pressure)
        except Exception as exc:
            print(f"[TTS] Synthesis error: {exc}")
        finally:
            _synth_queue.task_done()


def _play_worker() -> None:
    """Stage 2: play pre-synthesized audio chunks back-to-back."""
    while not _stop_event.is_set():
        try:
            item = _play_queue.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:
            _play_queue.task_done()
            break
        audio, sr = item
        _play_audio(audio, sr)
        _play_queue.task_done()


def _ensure_pipeline() -> None:
    global _pipeline_started
    if not _pipeline_started:
        with _pipeline_lock:
            if not _pipeline_started:
                _stop_event.clear()
                threading.Thread(target=_synth_worker, daemon=True).start()
                threading.Thread(target=_play_worker,  daemon=True).start()
                _pipeline_started = True


def clear_queue() -> None:
    """Discard queued sentences and stop playback immediately."""
    # Drain both queues
    for q in (_synth_queue, _play_queue):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break
    sd.stop()


def speak(text: str, voice_id: str | None = None) -> None:
    if not text.strip():
        return

    voice_id = resolve_voice_id(voice_id or default_voice_id())
    if not voice_id:
        print("[TTS] No voices configured. Add one in the UI.")
        return

    _ensure_pipeline()
    _synth_queue.put((text, voice_id))


def clean_text(text: str) -> str:
    """Public wrapper around ``_clean_for_tts`` for use outside this module."""
    return _clean_for_tts(text)

