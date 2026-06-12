"""Voice registry, manifest management, and reference-audio preparation.

Handles adding/removing voice profiles, converting audio to the required
format, Whisper-based reference transcription, and status tracking.
"""

import contextlib
import copy
import io
import json
import os
import re
import shutil
import subprocess
import uuid
import warnings

from config import BASE_DIR, TARGET_SR, VOICES_DIR, VOICES_MANIFEST

AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3", ".mpeg", ".mpg", ".mp4", ".ogg", ".flac", ".aac", ".wma")
CACHE_DIR = os.path.join(VOICES_DIR, ".cache")
WHISPER_MODEL = "openai/whisper-large-v3-turbo"


@contextlib.contextmanager
def _quiet_ml():
    """Silence Whisper / transformers stderr noise during transcription."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def _ensure_dirs() -> None:
    os.makedirs(VOICES_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def _abs_path(path: str) -> str:
    """Return an absolute path, resolving relative paths against BASE_DIR."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BASE_DIR, path))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or f"voice-{uuid.uuid4().hex[:8]}"


# In-memory manifest cache, invalidated by file mtime. synthesize() reads the
# manifest several times per streamed TTS chunk, so this keeps the hot path
# off the disk. Deep copies are returned/stored because callers mutate them.
_manifest_cache: dict | None = None
_manifest_mtime: float | None = None


def _load_manifest() -> dict:
    global _manifest_cache, _manifest_mtime
    _ensure_dirs()
    if not os.path.isfile(VOICES_MANIFEST):
        return {"voices": []}

    try:
        mtime = os.path.getmtime(VOICES_MANIFEST)
    except OSError:
        mtime = None
    if _manifest_cache is not None and mtime is not None and mtime == _manifest_mtime:
        return copy.deepcopy(_manifest_cache)

    try:
        with open(VOICES_MANIFEST, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"voices": []}

    if not isinstance(data, dict) or not isinstance(data.get("voices", []), list):
        return {"voices": []}

    _manifest_cache = copy.deepcopy(data)
    _manifest_mtime = mtime
    return data


def _save_manifest(data: dict) -> None:
    global _manifest_cache, _manifest_mtime
    _ensure_dirs()
    with open(VOICES_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _manifest_cache = copy.deepcopy(data)
    try:
        _manifest_mtime = os.path.getmtime(VOICES_MANIFEST)
    except OSError:
        _manifest_mtime = None


def _normalize_voice(voice: dict) -> dict:
    """Return a copy of *voice* with the file path resolved to an absolute path."""
    voice = dict(voice)
    voice["file"] = _abs_path(voice["file"])
    return voice


def list_voices() -> list[dict]:
    return [_normalize_voice(v) for v in _load_manifest().get("voices", [])]


def get_voice(voice_id: str) -> dict | None:
    for voice in list_voices():
        if voice["id"] == voice_id:
            return voice
    return None


def resolve_voice_id(selection: str | None) -> str:
    if not selection:
        return default_voice_id()

    if get_voice(selection):
        return selection

    lowered = selection.strip().lower()
    for voice in list_voices():
        if voice["name"].strip().lower() == lowered:
            return voice["id"]

    return selection


def voice_choices() -> list[tuple[str, str]]:
    """Return a list of (display_name, voice_id) tuples for UI dropdowns."""
    return [(v["name"], v["id"]) for v in list_voices()]


def default_voice_id() -> str:
    voices = list_voices()
    ready = [v for v in voices if v.get("status") == "ready"]
    if ready:
        return ready[0]["id"]
    return voices[0]["id"] if voices else ""


def voice_cache_path(voice_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{voice_id}.wav")


def preview_cache_path(voice_id: str) -> str:
    """Cached synthesized preview phrase, so repeat previews skip the GPU."""
    return os.path.join(CACHE_DIR, f"{voice_id}_preview.wav")


def voice_is_usable(voice_id: str) -> bool:
    voice = get_voice(voice_id)
    if voice is None:
        return False
    if os.path.isfile(voice_cache_path(voice_id)):
        return True
    try:
        _resolve_source_file(voice)
        return True
    except FileNotFoundError:
        return False


def _set_voice_status(voice_id: str, status: str, error: str = "") -> None:
    manifest = _load_manifest()
    for voice in manifest["voices"]:
        if voice["id"] == voice_id:
            voice["status"] = status
            if error:
                voice["error"] = error
            elif "error" in voice:
                del voice["error"]
            break
    _save_manifest(manifest)


def set_voice_ref_text(voice_id: str, ref_text: str) -> None:
    manifest = _load_manifest()
    for voice in manifest["voices"]:
        if voice["id"] == voice_id:
            voice["ref_text"] = ref_text.strip()
            break
    _save_manifest(manifest)


def get_voice_ref_text(voice_id: str) -> str:
    voice = get_voice(voice_id)
    if voice is None:
        return ""
    return voice.get("ref_text", "").strip()


def add_voice(name: str, source_path: str) -> str:
    _ensure_dirs()
    source_path = _abs_path(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Uploaded audio not found: {source_path}")

    voice_id = _slugify(name)
    manifest = _load_manifest()
    existing_ids = {v["id"] for v in manifest["voices"]}

    base_id = voice_id
    counter = 2
    while voice_id in existing_ids:
        voice_id = f"{base_id}-{counter}"
        counter += 1

    ext = os.path.splitext(source_path)[1].lower() or ".wav"
    dest_path = os.path.join(VOICES_DIR, f"{voice_id}{ext}")
    shutil.copy2(source_path, dest_path)

    manifest["voices"].append(
        {
            "id": voice_id,
            "name": name.strip(),
            "file": dest_path,
            "status": "processing",
        }
    )
    _save_manifest(manifest)
    _invalidate_cache(voice_id)
    return voice_id


def remove_voice(voice_id: str) -> bool:
    manifest = _load_manifest()
    voices = manifest.get("voices", [])
    target = next((v for v in voices if v["id"] == voice_id), None)
    if target is None:
        return False

    file_path = _abs_path(target["file"])
    if os.path.isfile(file_path):
        os.remove(file_path)

    manifest["voices"] = [v for v in voices if v["id"] != voice_id]
    _save_manifest(manifest)
    _invalidate_cache(voice_id)
    return True


def _invalidate_cache(voice_id: str) -> None:
    for cached in (voice_cache_path(voice_id), preview_cache_path(voice_id)):
        if os.path.isfile(cached):
            os.remove(cached)


def _resolve_source_file(voice: dict) -> str:
    source = _abs_path(voice["file"])
    if os.path.isfile(source):
        return source

    base, _ = os.path.splitext(source)
    for ext in AUDIO_EXTENSIONS:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate

    voice_dir = os.path.join(VOICES_DIR, voice["id"])
    for ext in AUDIO_EXTENSIONS:
        candidate = voice_dir + ext
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Voice sample missing for '{voice['name']}'. "
        f"Expected file at {source}. Re-add the voice in Voice Studio."
    )


def prepare_reference_audio(voice_id: str) -> str:
    voice = get_voice(voice_id)
    if voice is None:
        raise ValueError(f"Unknown voice: {voice_id}")

    _ensure_dirs()
    cached = voice_cache_path(voice_id)
    if os.path.isfile(cached):
        return cached

    source = _resolve_source_file(voice)
    if source.lower().endswith(".wav"):
        try:
            import soundfile as sf

            info = sf.info(source)
            if info.samplerate == TARGET_SR and info.channels == 1:
                shutil.copy2(source, cached)
                return cached
        except Exception:
            pass

    ffmpeg_exe = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            source,
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-sample_fmt",
            "s16",
            cached,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Audio conversion failed")

    return cached


_whisper_pipe = None


def _whisper_cached() -> bool:
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if not os.path.isdir(hub):
        return False
    return any("whisper-large-v3-turbo" in entry for entry in os.listdir(hub))


def _get_whisper_pipe():
    global _whisper_pipe
    if _whisper_pipe is not None:
        return _whisper_pipe

    import torch
    from transformers import pipeline

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).major >= 7
        else torch.float32
    )

    def _load(local_only: bool):
        kwargs = {"model_kwargs": {"local_files_only": True}} if local_only else {}
        return pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            torch_dtype=dtype,
            device=dev,
            **kwargs,
        )

    with _quiet_ml():
        try:
            if _whisper_cached():
                _whisper_pipe = _load(local_only=True)
            else:
                _whisper_pipe = _load(local_only=False)
        except Exception as local_err:
            if _whisper_cached():
                raise RuntimeError(
                    "Whisper model is cached but failed to load. "
                    f"Details: {local_err}"
                ) from local_err
            raise RuntimeError(
                "Could not download the Whisper model from Hugging Face (server may be busy). "
                "Paste a transcript in the optional field when adding a voice, or try again later."
            ) from local_err

    return _whisper_pipe


def transcribe_reference_text(voice_id: str, manual_text: str = "") -> str:
    """Transcribe the reference clip once and cache ref_text in the manifest."""
    if manual_text and manual_text.strip():
        set_voice_ref_text(voice_id, manual_text.strip())
        return manual_text.strip()

    existing = get_voice_ref_text(voice_id)
    if existing:
        return existing

    wav_path = prepare_reference_audio(voice_id)
    pipe = _get_whisper_pipe()

    with _quiet_ml():
        ref_text = pipe(
            wav_path,
            chunk_length_s=30,
            # 128 batched 30s chunks can OOM an 8 GB card that's also holding
            # the LLM and TTS; 16 is plenty for short reference/mic clips.
            batch_size=16,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()

    set_voice_ref_text(voice_id, ref_text)
    return ref_text


def transcribe_audio_file(source_path: str) -> str:
    """Transcribe an arbitrary audio file (e.g. a browser mic recording).

    Converts the input to 16 kHz mono WAV with FFmpeg first, then runs the
    same Whisper pipeline used for reference transcription — so mic input
    stays fully local and costs no extra VRAM.
    """
    source_path = _abs_path(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Audio not found: {source_path}")

    ffmpeg_exe = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    wav_path = os.path.join(CACHE_DIR, f"_mic_{uuid.uuid4().hex[:8]}.wav")
    _ensure_dirs()
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-i", source_path, "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", wav_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Audio conversion failed")

    try:
        pipe = _get_whisper_pipe()
        with _quiet_ml():
            text = pipe(
                wav_path,
                chunk_length_s=30,
                batch_size=16,  # same OOM consideration as transcribe_reference_text
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
        return text
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def is_voice_prepared(voice_id: str) -> bool:
    """True when ref_text exists and the source audio file is on disk."""
    voice = get_voice(voice_id)
    if voice is None or not voice.get("ref_text", "").strip():
        return False
    try:
        _resolve_source_file(voice)
        return True
    except FileNotFoundError:
        return False


def mark_voice_ready(voice_id: str) -> None:
    _set_voice_status(voice_id, "ready")


def mark_voice_failed(voice_id: str, error: str) -> None:
    _set_voice_status(voice_id, "error", error)


def validate_voices() -> None:
    """Reconcile each voice's manifest status with what is actually on disk.

    Marks voices as 'error' when their audio file is missing and restores
    them to 'ready' when a previously missing file reappears.
    """
    manifest = _load_manifest()
    changed = False

    for voice in manifest.get("voices", []):
        voice["file"] = _abs_path(voice["file"])
        usable = voice_is_usable(voice["id"])
        if voice.get("status") == "ready" and not usable:
            voice["status"] = "error"
            voice["error"] = "Audio file missing - re-add this voice"
            changed = True
        elif voice.get("status") == "error" and usable:
            voice["status"] = "ready"
            voice.pop("error", None)
            changed = True
        elif voice.get("status") == "processing" and usable and voice.get("ref_text", "").strip():
            voice["status"] = "ready"
            changed = True

    if changed:
        _save_manifest(manifest)
